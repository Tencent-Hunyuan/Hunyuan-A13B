import asyncio
import json
import traceback
import sys
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from openai import OpenAI


class RequestHandle:
    def __init__(self, mdoel_name, url):
        self.model_name = mdoel_name
        self.url = url

    def request(self, messages, tools=None):
        openai_api_base = f"{self.url}/v1"
        openai_api_key = "EMPTY"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        tools = tools
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            temperature=0.5,
            top_p=0.7,
            tools=tools,
            max_tokens=8192,
            extra_body={
                "repetition_penalty": 1.05,
                "top_k": 20
            },
        )
        response = response.model_dump()
        text = response["choices"][0]["message"]["content"]
        if "</think>" in text:
            think = text[text.find("<think>") + len("<think>"): text.rfind("</think>")].strip()
            response["choices"][0]["message"]["reasoning_content"] = think
        resp = None
        if "</answer>" in text:
            resp = text[text.find("<answer>") + len("<answer>"): text.rfind("</answer>")].strip()
            if resp.startswith("助手："):
                resp = resp[len("助手："):].strip()
        response["choices"][0]["message"]["content"] = resp
        return response


class MCPClient:
    def __init__(self, config):
        self.config = config
        self.mcp_stdio_server_handles = {}
        self.mcp_tools = {}
        self.tool2mcp = {}
        self.initialize()

    def initialize(self):
        return asyncio.run(self._initialize())

    async def _initialize(self):
        for mcp_name, mcp_config in self.config.items():
            server_params = StdioServerParameters(
                # 服务器执行的命令，这里我们使用 uv 来运行 web_search.py
                command = mcp_config["command"],
                # 运行的参数
                args = mcp_config["args"],
                env = mcp_config.get("env", None)
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.mcp_tools[mcp_name] = json.loads(tools.model_dump_json())["tools"]
                    self.mcp_stdio_server_handles[mcp_name] = server_params
            print(f"{mcp_name} initialized with {len(self.mcp_tools[mcp_name])} tools")
            for tool in self.mcp_tools[mcp_name]:
                if tool["name"] in self.tool2mcp:
                    print(
                        f"Warning! Tool {tool['name']} already exists in {self.tool2mcp[tool['name']]}, now {mcp_name} wanna add it")
                self.tool2mcp[tool["name"]] = mcp_name
        return self.mcp_tools

    def get_tools(self):
        if len(self.mcp_tools) > 0:
            return self.mcp_tools
        else:
            self.initialize()
            return self.mcp_tools

    async def _tool_call(self, tool_call):
        try:
            mcp_name = self.tool2mcp[tool_call["name"]]
            server_params = self.mcp_stdio_server_handles[mcp_name]
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tool_name = tool_call["name"]
                    tool_arguemnts = tool_call["arguments"]
                    if type(tool_arguemnts) == str:
                        tool_arguemnts = json.loads(tool_arguemnts)
                    elif type(tool_arguemnts) != dict:
                        raise Exception("toolcall的arguments必须是字典")
                    response = await session.call_tool(tool_name, tool_arguemnts)
                    response = response.model_dump_json()
        except:
            response = traceback.format_exc()
        return response

    def tool_call(self, tool_call):
        if tool_call["name"] not in self.tool2mcp:
            return f"Not found {tool_call['name']} tool in mcp server."
        if "name" not in tool_call or "arguments" not in tool_call:
            return "miss 'name' or 'arguments': " + json.dumps(tool_call, ensure_ascii=False)
        if type(tool_call) != dict:
            return "arguments should be dict type"
        return asyncio.run(self._tool_call(tool_call))


def main(request_handle):
    mcp_config = {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/"
            ]
        }
    }

    client = MCPClient(mcp_config)
    tools = []
    for server_name, server_tools in client.get_tools().items():
        for tool in server_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]

                }
            })
    print("---" * 5, "MCP 拥有以下工具", "---" * 5)
    print(json.dumps(tools, ensure_ascii=False, indent=2))

    try:
        reasons = []
        messages = []
        max_iter = 12
        while True:
            question = input("(请输入你的问题，输入exit退出) 用户：")
            if question.strip() == "exit":
                break
            messages.append({"role": "user", "content": question})
            for _ in range(max_iter):
                resp = request_handle.request(messages, tools)
                resp = resp["choices"][0]["message"]
                reason = resp["reasoning_content"] if "reasoning_content" in resp else None
                if reason is not None:
                    print("推理过程:\n" + reason.strip())
                    reasons.append(reason)
                messages.append({"role": "assistant", "content": resp.get("content", None),
                                 "tool_calls": resp.get("tool_calls", None)})
                if resp.get("tool_calls", None) is None or len(resp["tool_calls"]) == 0:
                    print("---" * 3)
                    print("助手：", resp["content"])
                    break
                else:
                    for tool_call in resp["tool_calls"]:
                        tool_call = tool_call["function"]
                        print("---" * 3 + "\n执行函数：\n" + json.dumps(tool_call, ensure_ascii=False, indent=2))
                        response = client.tool_call(tool_call)
                        print("执行结果：\n" + response)
                        messages.append({
                            "role": "tool",
                            "name": tool_call["name"],
                            "content": response
                        })
                countinue_flag = input("继续执行吗？(输入exit退出)")
                if countinue_flag.strip() == "exit":
                    break
    except:
        traceback.print_exc()


if __name__ == "__main__":
    handle = RequestHandle(sys.argv[1], sys.argv[2])
    main(handle)
