import random
import asyncio
import re

from sympy.codegen.ast import continue_


# 从路径的文本中依次读取各个网址
def read_urls_from_file(file_path):
    """从文本文件中读取URL列表"""
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls

# 贴吧: 爬取的数据预处理
# 1.判断输入文本是否在水经验 2.剔除无文字内容，仅有需展开图片的 (附：无文字但图片无需展开的在主程序if main_comment处已经剔除了) 3. 剔除评论和回复为非文本类型的，其识别为''的
    # 水经验: 返回match对象， 不水经验: 返回None
def pre_process_data(input_text, enable=True):
    if enable:
        # pattern正则表达式
            # 包含类似格式的正则表达式 ["3", "33", "333", "经验加3", "+3", "加3", "加三"]等
            # 包含无文字内容，仅有需展开图片的，此时文字为'点击展开，查看完整图片'
            # 图片富文本等，使得内容为''的回复
        pattern = r"^\s*(经验)?\s*(加|\+)?\s*(3+|三+)\s*(经验)?\s*$|^(点击展开，查看完整图片)$|^$"
        judge = re.search(pattern, input_text) # 若满足水经验正则表达式，则返回一个match对象
        return judge #
    return None

# 贴吧动态加载: 模仿人类移动网页，以加载动态网页同时防止被检测到爬虫
    # 附: 直接瞬移到底下，无法完成动态加载await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
async def human_like_scroll(page):
    current_y = 0
    final_y = await page.evaluate("document.body.scrollHeight");
    print(f"final_y:{final_y}")

    while True:
        step = step = random.randint(150, 400)
        current_y = current_y + step
        if current_y > final_y:
            current_y = final_y
        await page.evaluate(f"window.scrollTo(0, {current_y})")
        await page.wait_for_timeout(300)
        if current_y == final_y:
            break

# 百度贴吧 手机号密码自动登录
async def auto_login(page):
    need_for_auto_login = input("\n贴吧自动登录请输入a，其他手动登录 | Enter 'a' to auto login Tieba\n")
    if need_for_auto_login == "a":
        print("运行自动登录程序|Run the Tieba auto-login program")

        try:
            # 等待选择器匹配元素出现 等待初始登录弹窗出现
            await page.wait_for_selector('div.tieba-login-wrapper', state="visible")

            # 百度贴吧好像每次打开浏览器 其登录界面组件是动态id 同时其class类存在重复的，故下面用嵌套的类来定位组件
            # 采用账户密码输入方式
            await page.locator('div.tieba-login-wrapper p.tang-pass-footerBarULogin.pass-link').click()#
            # 输入用户名和密码
            await page.locator('div.tieba-login-wrapper input.pass-text-input.pass-text-input-userName').fill('15158552581') # 输入预设账户
            await page.locator('div.tieba-login-wrapper input.pass-text-input.pass-text-input-password').fill('abcdefg1234567') # 输入预设密码

            await asyncio.sleep(random.uniform(1, 3))

            # 点击登录按钮
            await page.locator('div.tieba-login-wrapper input.pass-button.pass-button-submit').click()
            print("自动登录成功")
        except:
            print("自动登录失败，请手动登录")
    else:
        print("请手动登录 | Please manual login")

    judge = input("\n已完成登录请输入1 | If Login complete, please enter 1\n")
    if judge==1:
        pass

    return 1

# 知乎基于评论数量的页面滚动加载
async def scroll_based_comments(page, max_answer_num):
    print(f"\n开始滚动加载页面，所需{max_answer_num}个回答 | Start loading the page, expecting {max_answer_num} answers")
    await page.wait_for_selector("div.List-item") # div.ContentItem.AnswerItem
    comment_reply = await page.locator("div.List-item").all()
    num = len(comment_reply)
    print(f"已加载{num}个回答 | {num} answers")
    current_y = 0
    max_answer_num = max_answer_num + 2 # 知乎在滚动网页动态加载时，有时会瞬间减少已刷新的回答数2-3个。比如刚开始刷新到了共10个，瞬间减少到8个。故这里加3，以防止需要爬10个问题时，才刷新了8个的情况
    count = 0

    while True:
        # 随机滚动长度
        step = step = random.randint(150, 400)
        current_y = current_y + step
        await page.evaluate(f"window.scrollTo(0, {current_y})")
        # 停一定时间，防止滚动过快被反爬检测到。
        await page.wait_for_timeout(300)

        # 更新已加载回答数
        comment_reply = await page.locator("div.List-item").all()
        last_num = num # 记录上一个已加载问题的数量
        num = len(comment_reply)
        if not (last_num < num):
            print(f"已加载{num}个回答 | {num} answers")

        # 加载到底部退出的情况
        bottom_button = page.locator("button.Button.QuestionAnswers-answerButton.FEfUrdfMIKpQDJDqkjte.Button--blue.Button--spread.JmYzaky7MEPMFcJDLNMG.GMKy5J1UWc7y8NF_V8YA:has-text('写回答')")
        if await bottom_button.count() > 0:
            print(f"已加载到底部无法进一步获取，共加载{num}个回答 | Reach the bottom, total load {num} answers")
            return comment_reply

        # 已加载回答数 > 目标回答数 退出
        if num >= max_answer_num:
            print(f"滚动加载完成，共加载{num}个回答 | Load completed, total load {num} answers")
            return comment_reply
  

# 测试登录
# async def main():
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=False)
#         context = await browser.new_context()
#         page = await context.new_page()
#
#         file_path = r"D:\pycharm_code\pythonProject\Web_Crawler\websites.txt"
#         urls = read_urls_from_file(file_path)
#         for url in urls:
#             await auto_login(page, url)
#
# if __name__ == "__main__":
#   asyncio.run(main())