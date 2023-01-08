class Question(object):
    def __init__(self, question_id, content):
        self.question_id = question_id
        self.content = content


class Questions(object):
    def __init__(self):
        self.questions = [
            Question(
                0, '你好！我是小I，很高兴认识你。可以请你做一个自我介绍吗？包括你对自己的描述，兴趣爱好，喜欢的事物等等，你可以先思考一下再开始。'
            ),
            Question(
                1, '谢谢你的介绍。我很好奇你觉得自己是内向的还是外向的呢？'
            ),
            Question(
                2, '可以举举生活中的例子具体说说吗？'
            ),
            Question(
                3, '好的，谢谢你的分享。你有过需要自己独自去学习一项完全没接触过的东西的经历吗？'
            ),
            Question(
                4, '你能大致描述一下当时是什么情况吗？以及是什么让你坚持下来的？'
            ),
            Question(
                5, '那请你试想一下，现在需要你学习一些对你来说完全没接触、很新的东西，你觉得你会去学吗？'
            ),
            Question(
                6, '好的，谢谢你的回答。你有过这样的经历吗？当你在忙自己的事情时，你的同学或朋友要你先把事情搁一边，帮助他/她完成一个对他来说非常非常重要的事情。'
            ),
            Question(
                7, '为什么愿意去学习呢？'
            ),
            Question(
                8, '为什么不会呢？什么样的原因才会让你去学习一些新东西呢？'
            ),
            Question(
                9, '你有放下手中的事情去帮助他吗？可以具体说说吗？'
            ),
            Question(
                10, '那请你试想一下这个场景，你会放下自己手中的事情去帮助他吗？'
            ),
            Question(
                11, '好的，谢谢你的回答。你最近或以前有过让你觉得特别不舒服或者感觉任务很困难的情况吗？'
            ),
            Question(
                12, '为什么呀？可以具体说说吗？'
            ),
            Question(
                13, '可以分享一下当时发生了什么吗？以及你是怎么处理这种情况的？'
            ),
            Question(
                14, '那请你试想一下，最近你的老师布置了一个特别花时间和精力的作业，而你最近又有其他事情必须要完成，时间和精力上都不太够，你会觉得很困难吗？'
            ),
            Question(
                15, '你会怎么处理这种情况呢？'
            ),
            Question(
                16, '为什么呀？是有什么好的经验吗？'
            ),
            Question(
                17, '好的，谢谢你的分享。请问你有做过时间线拉得比较长的任务吗？'
            ),
            Question(
                18, '你会提前做计划和安排吗？'
            ),
            Question(
                19, '那请你试想一下，如果现在有一个活动需要你策划和组织，你会提前做计划和安排吗？'
            ),
            Question(
                20, '可以具体分享一下你是怎么做计划和安排的吗？'
            ),
            Question(
                21, '你是怎么保证这项任务按时完成的呀？'
            ),
            Question(
                22, '好的，谢谢你的分享。以上就是我们所有的内容，再次感谢你能和我们交流。'
            )
        ]

    @staticmethod
    def get_next_question(cur_question_id: int, cur_answer:str):
        if cur_question_id == 0:
            return 1
        if cur_question_id == 1:
            if cur_answer.find('外向') != -1 or cur_answer.find('外') != -1:
                print('外向')
                return 2
            elif cur_answer.find('内向') != -1 or cur_answer.find('内') != -1:
                print('内向')
                return 2
            else:
                print('default 外向')
                return 2
        if cur_question_id == 2:
            return 3
        if cur_question_id == 3:
            if cur_answer.find('没有') != -1 or cur_answer.find('没') != -1:
                print('没有')
                return 5
            elif cur_answer.find('有') != -1:
                print('有')
                return 4
            else:
                print('default 有')
                return 4
        if cur_question_id == 4:
            return 6
        if cur_question_id == 5:
            if cur_answer.find('不会') != -1 or cur_answer.find('不') != -1:
                return 8
            elif cur_answer.find('会') != -1:
                return 7
            else:
                return 7
        if cur_question_id == 7 or cur_question_id == 8:
            return 6
        if cur_question_id == 6:
            if cur_answer.find('没有') != -1 or cur_answer.find('没') != -1:
                return 10
            elif cur_answer.find('有') != -1:
                return 9
            else:
                return 9
        if cur_question_id == 9:
            return 11
        if cur_question_id == 10:
            return 12
        if cur_question_id == 12:
            return 11
        if cur_question_id == 11:
            if cur_answer.find('没有') != -1 or cur_answer.find('没') != -1:
                return 14
            elif cur_answer.find('有') != -1:
                return 13
            else:
                return 13
        if cur_question_id == 13:
            return 17
        if cur_question_id == 14:
            if cur_answer.find('不会') != -1 or cur_answer.find('不太会') != -1 or cur_answer.find('不') != -1:
                return 16
            elif cur_answer.find('会') != -1:
                return 15
            else:
                return 15
        if cur_question_id == 15 or cur_question_id == 16:
            return 17
        if cur_question_id == 17:
            if cur_answer.find('没有') != -1 or cur_answer.find('没') != -1:
                return 19
            elif cur_answer.find('有') != -1:
                return 18
            else:
                return 18
        if cur_question_id == 18:
            return 20
        if cur_question_id == 19:
            return 21
        if cur_question_id == 20 or cur_question_id == 21:
            return 22
        print(cur_question_id, 'not hit any branch, return default: -2')
        return 22


if __name__ == '__main__':
    print('as'.find('d'))
    pass