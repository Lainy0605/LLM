# This is a sample Python script.
import pprint
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool


class Course:
    def __init__(self, name, type, credit):
        self.name = name
        self.type = type
        self.credit = credit


all_courses = [
    Course("数据结构与算法", "必修", 3),
    Course("Web前端开发", "选修", 2),
    Course("编译原理", "选修", 2),
    Course("计算机组织结构", "必修", 3),
    Course("离散数学", "必修", 2),
    Course("形势与政策", "必修", 0),
    Course("羽毛球", "选修", 1),
    Course("云计算", "选修", 2),
]
chosen_courses = []


@register_tool('select_chosen_courses')
class SelectAll(BaseTool):
    description = '返回用户当前已选的课程'
    parameters = []

    def call(self, params: str, **kwargs) -> list:
        return chosen_courses


@register_tool("select")
class Select(BaseTool):
    description = "查询所有可选课程"
    parameters = [{
        'name': "type",
        'type': "string",
        'description': '课程的类型，分为选修和必修',
        'enum': ["选修", "必修"],
        'required': False
    }]

    def call(self, params: str, **kwargs) -> list:
        type = json5.loads(params)['type']
        print(type)
        if type is None:
            return all_courses
        else:
            result = []
            for course in all_courses:
                if course.type == type:
                    result.append(course)
            return result


# 步骤 1（可选）：添加一个名为 `choose` 的自定义工具。
@register_tool('choose')
class Choose(BaseTool):
    description = '选择课程服务，如果返回值为1，给用户返回当前已选的课程，如果返回值为0，智能从所有可选课程中返回给用户可能想要选择的课程，如果返回值为2，告诉用户当前课程他已经选择过了'
    parameters = [{
        'name': 'course',
        'type': 'string',
        'description': '用户想要选择的课程，应该从所有课程中选择最为匹配的那个',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> int:
        course_name = json5.loads(params)['course']
        for course in chosen_courses:
            if course.name == course_name:
                return 2
        for course in all_courses:
            if course.name == course_name:
                chosen_courses.append(course)
                return 1
        return 0


@register_tool("delete")
class Delete(BaseTool):
    description = "删除某个已选课程，如果返回值为假，智能从用户当前已选的课程中返回给用户可能想要删除的课程，并向用户确认，如果返回值为真，返回给用户当前已选的课程"
    parameters = [{
        'name': 'course',
        'type': 'string',
        'description': '用户想要删除的课程名称',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> bool:
        course_name = json5.loads(params)['course']
        for course in chosen_courses:
            if course.name == course_name:
                return True
        return False


def init_agent_service():
    # 步骤 2：配置您所使用的 LLM。
    llm_cfg = {
        'model': 'Qwen2.5-14B',
        'model_server': 'http://10.58.0.2:8000/v1',
        'api_key': 'None',
    }

    # 步骤 3：创建一个智能体。这里我们以 `Assistant` 智能体为例，它能够使用工具并读取文件。
    system_instruction = '''你是一个选课系统的AI助手。
    在收到用户的请求后，你可以：
    - 1. 查询: 查询当前所有的课程，可以筛选必修或者选修，你应该根据描述将用户最感兴趣的课程放在前面，例如：用户喜欢体育，羽毛球等放在前面
    - 2. 选课：选择需要的课程，智能返回结果(成功返回选课结果, 未成功返回错误)
    - 3. 删除：删除选择的课程，智能返回结果。
    你总是用中文回复用户。'''

    tools = ['select_chosen_courses', 'select', 'choose', 'delete']
    b = Assistant(llm=llm_cfg,
                  name='选课系统助手',
                  description="帮助用户智能选课",
                  system_message=system_instruction,
                  function_list=tools)

    return b


if __name__ == '__main__':
    # 步骤 4：作为聊天机器人运行智能体。
    bot = init_agent_service()
    messages = []  # 这里储存聊天历史。
    while True:
        # 例如，输入请求 "绘制一只狗并将其旋转 90 度"。
        query = input('用户请求: ')
        # 将用户请求添加到聊天历史。
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run_nonstream(messages=messages, stream=False):
            # 流式输出。
            print('机器人回应：')
            print(response)
            # pprint.pprint(response, indent=2)
        # 将机器人的回应添加到聊天历史。
        messages.extend(response)
