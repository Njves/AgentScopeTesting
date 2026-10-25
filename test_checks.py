import unittest

from download_articles import extract
from main import Answer, Questions, check_ids, parse_json, quote_exists


class Checks(unittest.TestCase):
    def test_quotes(self):
        self.assertTrue(quote_exists("занимают место", "Индексы занимают\nместо."))
        self.assertFalse(quote_exists("не занимают место", "Индексы занимают место."))
        self.assertFalse(quote_exists("", "Индексы занимают место."))

    def test_ids(self):
        item = Answer(id=1, answer="ответ", quote="цитата")
        check_ids([item], [1])
        with self.assertRaises(ValueError):
            check_ids([item, item], [1, 2])

    def test_json(self):
        value = parse_json('```json\n{"questions":["a","b","c","d","e"]}\n```', Questions)
        self.assertEqual(len(value.questions), 5)
        with self.assertRaises(ValueError):
            parse_json('{"questions":["a"]}', Questions)

    def test_article_body_only(self):
        title, text = extract('<h1>Тема</h1><nav>Меню</nav><div class="article-formatted-body"><p>Лекция</p><script>bad()</script></div><div>Комментарии</div>')
        self.assertEqual((title, text), ("Тема", "Лекция"))
        with self.assertRaises(ValueError):
            extract("<h1>Access denied</h1>")


if __name__ == "__main__":
    unittest.main()
