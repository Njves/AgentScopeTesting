"""Download only the article body, without comments or navigation."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

ARTICLES = [
    "https://habr.com/ru/companies/ruvds/articles/724066/",
    "https://habr.com/ru/articles/900440/",
]
ROOT = Path(__file__).resolve().parent


def extract(html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("h1")
    body = soup.select_one(".article-formatted-body")
    if title is None or body is None:
        raise ValueError("Не найден текст статьи: возможно, Хабр вернул страницу блокировки")
    for element in body.select("script, style"):
        element.decompose()
    return title.get_text(" ", strip=True), body.get_text("\n", strip=True)


def main():
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    for url in ARTICLES:
        request = Request(url, headers={"User-Agent": "AgentScope-learning-demo/1.0"})
        with urlopen(request, timeout=30) as response:
            title, text = extract(response.read().decode("utf-8"))
        if len(text) < 1000:
            raise ValueError(f"Подозрительно короткая статья: {url}")
        path = directory / f"{url.rstrip('/').split('/')[-1]}.json"
        path.write_text(json.dumps(dict(url=url, title=title, text=text), ensure_ascii=False, indent=2))
        print(f"{path.name}: {title}, {len(text)} символов")


if __name__ == "__main__":
    main()
