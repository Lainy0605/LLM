# Reference: https://platform.openai.com/docs/guides/function-calling
import json
from qwen_agent.llm import get_chat_model


class Course:
    def __init__(self, name, type, credit):
        self.name = name
        self.type = type
        self.credit = credit

    def to_json(self):
        temp = {'name': self.name, 'type': self.type, 'credit': self.credit}
        return temp


all_courses = [
    Course("数据结构与算法", "必修", '3'),
    Course("Web前端开发", "选修", '2'),
    Course("编译原理", "选修", '2'),
    Course("大数据开发", "必修", '3'),
    Course("离散数学", "必修", '2'),
    Course("线形代数", "必修", '3'),
    Course("微积分", "必修", '4'),
    Course("体育", "选修", '1'),
    Course("云计算", "选修", '2'),
]
chosen_courses = []


class SelectChosenCourses:
    def call(self, params: str) -> str:
        results = []
        for course in chosen_courses:
            results.append(course.to_json())
        return json.dumps(results)


class SelectAllCourses:
    def call(self, params: str) -> str:
        type = params['type']
        results = []
        if type == "全部":
            for course in all_courses:
                results.append(course.to_json())
            return json.dumps(results)
        else:
            for course in all_courses:
                if course.type == type:
                    results.append(course.to_json())
            return json.dumps(results)


class ChooseCourse:
    def call(self, params: str) -> str:
        course_name = params['course']
        for course in chosen_courses:
            if course.name == course_name:
                return '这门课已经选择过了'
        for course in all_courses:
            if course.name == course_name:
                chosen_courses.append(course)
                return '选择成功'
        return '系统中没有这门课'


class DeleteCourse:
    def call(self, params: str) -> str:
        course_name = params['course']
        for course in chosen_courses:
            if course.name == course_name:
                chosen_courses.remove(course)
                return '删除成功'
        for course in all_courses:
            if course.name == course_name:
                return '这门课没有选过'
        return '系统中没有这门课'


def init_agent_service():
    llm = get_chat_model({
        'model': 'Qwen2.5-14B',
        'model_server': 'http://10.58.0.2:8000/v1',
        'api_key': 'None',
    })

    return llm


def get_functions():
    functions = [
        {
            'name': 'select_chosen_courses',
            'description': '查询用户已经选择的课程',
            'parameters': {},
        },
        {
            'name': 'select_all_courses',
            'description': '根据课程类型查询当前系统中的所有课程',
            'parameters': {
                'type': 'object',
                'properties': {
                    'type': {
                        'type': 'string',
                        'description': '课程类型',
                        'enum': ['选修', '必修', '全部']
                    }
                },
            }
        },
        {
            'name': 'choose_course',
            'description': '选择课程服务，根据用户提供的课程名帮用户选择课程，结果可能是选择成功、这门课已经选择过了或者系统中没有这门课',
            'parameters': {
                'type': 'object',
                'properties': {
                    'course': {
                        'type': 'string',
                        'description': '用户想要选择的课程'
                    }
                },
                'required': ['course'],
            }
        },
        {
            'name': 'delete_course',
            'description': '删除课程服务，根据用户提供的课程帮用户删除课程，结果可能是删除成功、这门课没有选过或者系统中没有这门课',
            'parameters': {
                'type': 'object',
                'properties': {
                    'course': {
                        'type': 'string',
                        'description': '用户想要删除的课程'
                    }
                },
                'required': ['course']
            }
        }
    ]

    return functions


def call_function(response):
    available_functions = {
        'select_chosen_courses': SelectChosenCourses(),
        'select_all_courses': SelectAllCourses(),
        'choose_course': ChooseCourse(),
        'delete_course': DeleteCourse(),
    }

    function_name = response['function_call']['name']
    function_to_call = available_functions[function_name]
    function_args = json.loads(response['function_call']['arguments'])
    function_response = function_to_call.call(params=function_args)

    return function_name, function_response


def handle_response(messages):
    responses = llm.chat(
        messages=messages,
        functions=get_functions(),
        stream=False,
    )
    messages.extend(responses)
    for response in responses:
        if response.get('function_call', None):
            function_name, function_response = call_function(response)
            messages.append({
                'role': 'function',
                'name': function_name,
                'content': function_response
            })
            handle_response(messages)
        else:
            print(response['content'])


if __name__ == '__main__':
    llm = init_agent_service()
    system_instruction = '''你是一个选课系统的AI助手。
    在收到用户的请求后，你可以：
    - 1. 查询系统中所有课程: 根据课程类型查询系统中的所有课程，并且你应该根据描述将用户最感兴趣的课程放在前面
    - 2. 查询用户已选课程：查询用户已经选择的课程
    - 3. 选择课程：选择用户给定的课程。如果选择成功你应该给用户展示已选课程，如果这门课已经选择过了就告诉用户，如果系统中没有这门课就从所有课程中给用户推荐最可能感兴趣的课程
    - 4. 删除课程：删除用户给定的课程。如果选择成功你应该给用户展示已选课程，如果这门课没有选过就告诉用户，如果系统中没有这门课就从用户已经选择的课程中给用户推荐给最可能想要删除的课程
    你总是用中文回复用户。请注意由于课程系统可能会随时更新，所以你应当总是通过函数调用的方式获取实时精确的课程信息，而不是从上下文信息中推断课程信息'''
    messages = [{'role': 'system', 'content': system_instruction}, {'role': 'user', 'content': '介绍你自己'}]
    handle_response(messages)
    while True:
        query = input("请输入: ")
        messages.append({'role': 'user', 'content': query})
        handle_response(messages)
