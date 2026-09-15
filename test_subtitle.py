"""Unit tests for the parts of subtitle.py that need neither ffmpeg nor Whisper.

Run with:  uv run python -m unittest
"""

import unittest

from subtitle import (CJK_LAYOUT, DEFAULT_LAYOUT, MIN_CUE_SECONDS, Word, build_cues,
                      layout_for, srt_time, wrap_lines)


def words(text: str, start: float = 0.0, step: float = 0.3) -> list[Word]:
    """Turn a sentence into evenly spaced Words, one per whitespace-separated token."""
    out = []
    t = start
    for token in text.split():
        out.append(Word(t, t + step, token))
        t += step
    return out


class SrtTimeTest(unittest.TestCase):
    def test_formats_hours_minutes_seconds_millis(self):
        self.assertEqual(srt_time(0), "00:00:00,000")
        self.assertEqual(srt_time(3723.456), "01:02:03,456")

    def test_rounds_to_nearest_millisecond(self):
        self.assertEqual(srt_time(1.0006), "00:00:01,001")


class LayoutTest(unittest.TestCase):
    def test_known_and_unknown_languages(self):
        self.assertEqual(layout_for("ja"), CJK_LAYOUT)
        self.assertEqual(layout_for("ru"), DEFAULT_LAYOUT)
        self.assertEqual(layout_for("xx"), DEFAULT_LAYOUT)


class BuildCuesTest(unittest.TestCase):
    def test_short_speech_is_one_cue(self):
        cues = build_cues(words("Привет, как дела?"), DEFAULT_LAYOUT)
        self.assertEqual([c.text for c in cues], ["Привет, как дела?"])

    def test_splits_on_long_pause(self):
        first = words("Первая фраза.")
        second = words("Вторая фраза.", start=first[-1].end + 2.0)
        cues = build_cues(first + second, DEFAULT_LAYOUT)
        self.assertEqual([c.text for c in cues], ["Первая фраза.", "Вторая фраза."])

    def test_splits_after_sentence_end(self):
        cues = build_cues(words("Это первое предложение. А это второе."), DEFAULT_LAYOUT)
        self.assertEqual([c.text for c in cues],
                         ["Это первое предложение.", "А это второе."])

    def test_never_exceeds_two_lines_worth_of_text(self):
        text = " ".join(["слово"] * 60)
        cues = build_cues(words(text), DEFAULT_LAYOUT)
        limit = DEFAULT_LAYOUT.max_line_chars * 2
        self.assertTrue(all(len(c.text) <= limit for c in cues))
        self.assertEqual(" ".join(c.text for c in cues), text)

    def test_forced_split_backs_up_to_punctuation(self):
        text = "Один два три четыре пять, шесть семь восемь девять десять одиннадцать двенадцать тринадцать"
        cues = build_cues(words(text), DEFAULT_LAYOUT)
        self.assertEqual(cues[0].text, "Один два три четыре пять,")

    def test_joins_without_spaces_for_cjk(self):
        cues = build_cues(words("今日 は いい 天気 です"), CJK_LAYOUT)
        self.assertEqual(cues[0].text, "今日はいい天気です")

    def test_minimum_duration_and_no_overlap(self):
        # Two very short cues separated by a pause: the first is stretched
        # towards MIN_CUE_SECONDS but must still end before the second starts.
        first = [Word(0.0, 0.1, "Да.")]
        second = [Word(1.0, 1.1, "Нет.")]
        cues = build_cues(first + second, DEFAULT_LAYOUT)
        self.assertEqual(len(cues), 2)
        self.assertLess(cues[0].end, cues[1].start)
        self.assertGreaterEqual(cues[1].end - cues[1].start, MIN_CUE_SECONDS)


class WrapLinesTest(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(wrap_lines("короткая строка", DEFAULT_LAYOUT), "короткая строка")

    def test_long_text_becomes_two_balanced_lines(self):
        text = "это довольно длинная реплика которая точно не помещается в одну строку"
        lines = wrap_lines(text, DEFAULT_LAYOUT).split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= DEFAULT_LAYOUT.max_line_chars for line in lines))
        self.assertEqual(" ".join(lines), text)

    def test_prefers_break_after_punctuation(self):
        text = "Простите, что беспокою вас, но дело очень важное."
        lines = wrap_lines(text, DEFAULT_LAYOUT).split("\n")
        self.assertTrue(lines[0].endswith(","))

    def test_cjk_breaks_between_characters(self):
        text = "今日はとてもいい天気ですね、散歩に行きましょう"   # 24 chars > 18 per line
        lines = wrap_lines(text, CJK_LAYOUT).split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual("".join(lines), text)


if __name__ == "__main__":
    unittest.main()
