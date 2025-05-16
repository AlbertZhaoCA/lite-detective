import asyncio
import json
import random
import re
import sys

import pandas as pd
from datashader import first
from lazy_object_proxy.utils import await_
from playwright.async_api import async_playwright
from Assistant_Methods import read_urls_from_file
from Assistant_Methods import scroll_based_comments
from Assistant_Methods import auto_login

async def scrape_zhihu_comments(url_list, max_answers_num):
    comments_data = []

    async with async_playwright() as p:
        # 爬虫设置
        type = input("使用新的google浏览器 请输入1 | Using new google browser, enter coldet_baseline\n" "使用本地google浏览器 请输入2 | Using local google browser, enter 2\n")
        if type == "coldet_baseline":
            browser = await p.chromium.launch(headless=False, # 可设为 False 调试
            args = ["--disable-blink-features=AutomationControlled"], # 隐藏webdriver特征
            ignore_default_args=["--enable-automation"]
            )

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
            )
        elif type == "2":
            context = await p.chromium.launch_persistent_context(
                user_data_dir=r"C:\Users\Hu LuLu\AppData\Local\Google\Chrome\playwrightProfile",
                headless=False,
                executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # 指定你的 chrome 路径
                args=["--profile-directory=Default", # 指定使用的配置文件，比如 上面user_data_dir下的Default文件夹
                      "--disable-blink-features=AutomationControlled",  # 隐藏webdriver特征
                      ],
                ignore_default_args=["--enable-automation"]
            )
        else:
            print("输入错误，未能成功初始化context")
        page = await context.new_page()

        success_login = 0
        # 通过网址遍历每个知乎主题
        for post_url in url_list:
            try:
                # 进行网页登录操作
                if success_login == 0:  # 未成功登录
                    # wait_until="commit" 当浏览器接受到http响应头时，立即直接运行后续代码，不用检测该页面是否加载完成。防止出现timeout未加载完报错
                    # 百度的弹窗登录机制：在登录弹窗未关闭或者成功登录前，页面会一直处于未加载完成状态，故无commit的goto定会报错
                    await page.goto(post_url, wait_until="commit")
                    await asyncio.sleep(random.uniform(1, 3))  # 等待弹窗加载并出现
                    success_login = await auto_login(page)
                    await asyncio.sleep(random.uniform(2, 5))
                elif success_login == 1: # 已成功登录
                    await page.goto(post_url)
                    await asyncio.sleep(random.uniform(2, 5))
                print(f"\n正在抓取帖子: {post_url} | Currently crawling: {post_url}")

                # 获取网页标题 并移除非法字符
                title = await page.title()
                safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)

                # 获取该问题的主题
                question = page.locator("h1.QuestionHeader-title").first
                question_text = (await question.inner_text()).strip()

                # 获取该问题详情
                detail_button = page.locator("button.Button.QuestionRichText-more.FEfUrdfMIKpQDJDqkjte.Button--plain.fEPKGkUK5jyc4fUuT0QP").first
                if await detail_button.count() > 0:
                    print("已成功爬取该问题详情 | Already crawled description")
                    await detail_button.click()
                    # 获取详情信息
                    detail = page.locator("span.RichText.ztext.css-1yl6ec1").first
                    detail_text = (await detail.inner_text()).strip()
                else:
                    print("该问题没有详细描述 | No description")
                    detail_text = ""

                # 判定 全部回答按钮是否存在，若存在则点击
                print("\n'查看全部回答'按钮是否存在 | Check whether 'All Answers' button exists")
                all_comments_button = page.locator("div.Card.ViewAll a.QuestionMainAction.ViewAll-QuestionMainAction").first
                if await all_comments_button.count() == 1:
                    await all_comments_button.click()
                    print("\t存在，已自动点击 | Exists, already clicked button" )
                else:
                    print("\t不存在，已加载全部内容 | Not exist")

                # 滚动网页, 根据所需的最大回答数，
                comment_reply_elements= await scroll_based_comments(page, max_answers_num)
                await page.wait_for_timeout(2000) # 等待加载完全

                answers_num = 0 # 当前爬的是第几个回答
                window_exist = 0 # 回复弹窗是否存在 0默认不存在
                for comment_reply in comment_reply_elements: # 遍历该问题的每个回答
                    # 当前正在爬取的回答的编号
                    if answers_num == max_answers_num:
                        print(f"\n设定爬取{max_answers_num}个回答，已完成爬取{answers_num}个回答，任务已完成 | Mission complete")
                        break
                    answers_num += 1
                    print(f"\n正在爬取第{answers_num}个回答 | Answer {answers_num}")

                    # 获取层主评论
                    comment = comment_reply.locator("div.css-376mun").first  # comment尽管只有一个，但.all()输出的是一个数组，数组是没有.inner_text()方法的。只需第一个.first (属性非方法)
                    main_comment = (await comment.inner_text()).strip()

                    # 定位评论按钮并点击, 以展开所有评论
                    button = comment_reply.locator("button.Button.ContentItem-action:has-text('条评论')").first
                    if await button.count() > 0: # 不能等于1，因为该按钮有2个
                        await button.click()
                        print("\t(评论按钮已点击", end=' ')
                    else:
                        print("\t无评论或评论区已关闭 | No comments or comment area is closed")
                        # 存储该回答信息
                        if main_comment:
                            comments_data.append({
                                "floor_number": answers_num,
                                "comment": main_comment,
                                "replys": []
                            })
                            print(f"第{answers_num}个回答已完成爬取 | Answer {answers_num} complete")
                        # 跳出当前循环，爬取下一回答
                        continue

                    # 判断评论弹窗是否存在
                    reply_window = page.locator("div.Modal-content.css-1svde17").first
                    if await reply_window.count() > 0:
                        comment_reply = reply_window # comment_reply为弹窗位置
                        window_exist = 1
                        print("以弹窗形式存在) | (Comments in popup)")
                    else:
                        window_exist = 0 # comment_reply为默认的每层位置
                        print("以非弹窗形式存在) | (Comments not in popup)")

                    # 判定评论区是否关闭
                    await page.wait_for_timeout(500)  # 留时间给评论区加载，否则程序运行速度往往快于评论区加载速度，导致"评论区已关闭"无法被检测到
                    reply_close = comment_reply.locator("text=评论区已关闭") # 或者"div.css-189h5o3"
                    if await reply_close.count() > 0:
                        print("\t评论区已关闭 | The comment area is closed")
                        reply_comments = []
                    else: # 评论区未关闭，收集评论

                        # 定位所有回复的区域
                        all_reply_area = comment_reply.locator("div.css-18ld3w0")
                        await all_reply_area.locator('div[data-id]').first.wait_for()  # 要等待评论加载出来，程序运行到下一步的速度肯定是比评论加载速度快，导致下一步获取评论时为空[]
                        all_replys = await all_reply_area.locator('>div[data-id]').all()  # 加上>, 以仅匹配all_reply_area的子集中复合div[data-id]的元素，但不包含子集以下的
                        replys_num = len(all_replys)

                        # 获取所有的回复
                        # 存所有回复         主回复            所有主回复的回复              所有主回复的回复         由单个主回复的回复组成
                        # reply_comments[main_reply_text, replys_to_main_reply]  其中 replys_to_main_reply[each_secondary_reply_text]
                        reply_comments = []  # 用于存储该回答下的 主回复，和回复的回复
                        for each_reply in all_replys:
                            # 定位并获取主回复
                            main_reply = each_reply.locator("div.css-jp43l4 div.CommentContent.css-1jpzztt").first
                            main_reply_text = (await main_reply.inner_text()).strip()
                            reply_comments.append(main_reply_text)  # 存入主回复

                            # 定位获取对主回复的回复
                            replys_to_main_reply = []
                            secondary_replys = await each_reply.locator("div[data-id]").all()
                            for each_secondary_reply in secondary_replys:
                                each_secondary_reply = each_secondary_reply.locator(
                                    "div.CommentContent.css-1jpzztt").first
                                each_secondary_reply_text = (await each_secondary_reply.inner_text()).strip()
                                replys_to_main_reply.append(each_secondary_reply_text)  # 依次存入单个主回复的回复
                            reply_comments.append(replys_to_main_reply)  # 存入所有主回复的回复的列表
                        print(f"\t已成功爬取回答，及其{replys_num}个回复 | {replys_num} comments")

                    # 若为弹窗模式显示回复，实现关闭弹窗
                    if window_exist == 1:
                        close_button = page.locator("div.css-1aq8hf9 button[aria-label = '关闭']").first
                        await close_button.click()
                        window_exist = 0
                        print("\t(弹窗存在，已关闭弹窗) | (Close popup)")
                    else:
                        print("\t(弹窗不存在，无需关闭弹窗) | (No popup to close)")

                    # 整合存储以上信息
                    if main_comment:
                        comments_data.append({
                            "floor_number": answers_num,
                            "comment": main_comment,
                            "replys": reply_comments
                        })
                        print(f"第{answers_num}个回答已完成爬取 | Answer {answers_num} complete")

                    await asyncio.sleep(1)

            except Exception as e:
                print(f"抓取第 {answers_num} 个回答失败: {e}")
                # 无需break，break则已爬的就不会保存了

            # 单个帖子已读取完后，保存为 CSV
            with open(f"data/{safe_title}.csv", "w", encoding="utf-8-sig") as file:
                file.write(f"# 网址: {post_url}\n")
                # file.write(f"# 问题: {question_text}\n") # 写入问题标题
                # file.write((f"# 问题详情: {detail_text}\n")) # 写入问题的详情

            df = pd.DataFrame(comments_data)
            df.to_csv(f"data/{safe_title}.csv", mode='a', index=False, encoding="utf-8-sig")
            print(f"CSV文件已生成：{safe_title}.csv | Generate CSV file")

            # 列表清空以存储下一帖子内容
            comments_data = []

        # 随机等待降低风险 在爬不同帖子间
        await asyncio.sleep(random.uniform(2,4))

        close = input("任务已结束，关闭浏览器请输入1 | Enter coldet_baseline to close browser\n")
        if close == 1:
            await context.close()


# 运行主函数
if __name__ == "__main__":
    file_path = r"D:\pycharm_code\pythonProject\Web_Crawler\websites.txt"
    urls = read_urls_from_file(file_path)
    asyncio.run(scrape_zhihu_comments(url_list=urls, max_answers_num=15))
