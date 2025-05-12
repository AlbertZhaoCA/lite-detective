import asyncio
import json
import random
import re
import pandas as pd
from playwright.async_api import async_playwright
from sqlalchemy.sql.operators import all_op

from Assistant_Methods import read_urls_from_file
from Assistant_Methods import human_like_scroll
from Assistant_Methods import auto_login

async def scrape_tieba_comments(url_list, max_pages_per_post):
    comments_data = []

    async with async_playwright() as p:
        # 爬虫设置
        type = input("使用新的google浏览器 请输入1 | Using new google browser, enter 1\n" "使用本地google浏览器 请输入2 | Using local google browser, enter 2\n")
        if type == "1":
            browser = await p.chromium.launch(headless=False)  # 可设为 False 调试
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
            )
        elif type == "2":
            context = await p.chromium.launch_persistent_context(
                user_data_dir=r"C:\Users\Hu LuLu\AppData\Local\Google\Chrome\playwrightProfile",
                headless=False,
                executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # 指定你的 chrome 路径
                args=["--profile-directory=Default",  # 指定使用的配置文件，比如 上面user_data_dir下的Default文件夹
                      "--disable-blink-features=AutomationControlled",  # 隐藏webdriver特征
                      ],
                ignore_default_args=["--enable-automation"]
            )
        else:
            print("输入错误，未能成功初始化context")
        page = await context.new_page()

        success_login = 0
        # 遍历每个帖子
        for post_base_url in url_list:
            await page.goto(post_base_url)
            print(f"\n正在抓取帖子: {post_base_url} | Crawling post: {post_base_url}")

            # 读取当前帖子的总页数，
            all_page_num = page.locator("ul.l_posts_num li.l_reply_num span.red").nth(1)
            all_page_num = (await all_page_num.inner_text()).strip()
            all_page_num = int(all_page_num)
            print(f"共{all_page_num}页 | Total {all_page_num} pages")

            for page_num in range(1, max_pages_per_post + 1):
                post_url = f"{post_base_url}?pn={page_num}"
                try:
                    # 进行网页登录操作
                    if success_login == 0:  # 进行首次登录 或者 未成功登录
                        # wait_until="commit" 当浏览器接受到http响应头时，立即直接运行后续代码，不用检测该页面是否加载完成。防止出现timeout未加载完报错
                        # 百度的弹窗登录机制：在登录弹窗未关闭或者成功登录前，页面会一直处于未加载完成状态，故无commit的goto定会报错
                        await page.goto(post_url, wait_until="commit")
                        await asyncio.sleep(random.uniform(1, 3))  # 等待弹窗加载并出现
                        success_login = await auto_login(page)
                        await asyncio.sleep(random.uniform(2, 5))

                    elif success_login == 1: # 已成功登录
                        await page.goto(post_url)
                        await asyncio.sleep(random.uniform(2, 5))

                    # 若当前页数 > 当前帖子总页数 则跳出
                    if page_num > all_page_num:
                        print(f"第 {page_num} 页无评论，跳出 | The {page_num} has no comments, exit")
                        break

                    await human_like_scroll(page)
                    await page.wait_for_timeout(2000) # 等待加载完全

                    title = await page.title()
                    safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)

                    comment_reply_elements = await page.locator("div.p_postlist > div.l_post").all() # 仅div.p_postlist下一级的div.l_post是帖子回复综合体。其孙子集中还有一个div.l_post是广告会报错
                    for comment_reply in comment_reply_elements: # 遍历该页的每层(评论和回复综合体)
                        # 获取层主评论
                        comment = comment_reply.locator("div.d_post_content").first  # comment尽管只有一个，但.all()输出的是一个数组，数组是没有.inner_text()方法的。只需第一个.first (属性非方法)
                        main_comment = (await comment.inner_text()).strip()

                        # 获取层数信息
                        floor_num = None
                        info = comment_reply.locator("div.j_lzl_container").first
                        data_json = await info.get_attribute("data-field")  # 获取 data-field 属性值（JSON字符串）
                        if data_json: # 解析 JSON
                            data = json.loads(data_json)
                            floor_num = data.get("floor_num")  # 层数

                        # 获取该层的回复评论
                        reply_comments = []
                        replys = await comment_reply.locator("span.lzl_content_main").all()
                        if replys:
                            for reply in replys:
                                rep = (await reply.inner_text()).strip()
                                reply_comments.append(rep)

                        # 整合以上信息
                        if main_comment:
                            comments_data.append({
                                "floor_number": floor_num,
                                "comment": main_comment,
                                "replys": reply_comments
                            })
                            print(f"第{floor_num}层已完成爬取 |  Floor {floor_num} crawling completed")

                    print(f"第 {page_num} 页完成，抓到 {len(comment_reply_elements)} 条评论 | Page {page_num} completed, total {len(comment_reply_elements)} comments")

                except Exception as e:
                    print(f"抓取第 {page_num} 页失败: {e}")
                    break

            # 单个帖子已读取完后，保存为 CSV
            with open(f"data/{safe_title}.csv", "w", encoding="utf-8-sig") as file:
                file.write(f"# 网址: {post_base_url}\n")
                file.write(f"# 名称: {safe_title}\n")

            df = pd.DataFrame(comments_data)
            df.to_csv(f"data/{safe_title}.csv", mode='a', index=False, encoding="utf-8-sig")
            print(f"CSV文件已生成：{safe_title}.csv | Generate CSV file")

            # 列表清空以存储下一帖子内容
            comments_data = []

            # 随机等待降低风险
            await asyncio.sleep(random.uniform(2, 4))

        close = input("\n任务已结束，关闭浏览器请输入1 | Task complete, close browser please enter 1\n")
        if close == 1:
            await context.close()


# 运行主函数
if __name__ == "__main__":
    file_path = r"D:\pycharm_code\pythonProject\Web_Crawler\websites.txt"
    urls = read_urls_from_file(file_path)
    asyncio.run(scrape_tieba_comments(url_list=urls, max_pages_per_post=2))
