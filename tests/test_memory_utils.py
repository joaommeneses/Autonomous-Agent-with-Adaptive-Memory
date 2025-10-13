import unittest

from eval_utils import parse_episode


class TestParseEpisode(unittest.TestCase):

    def test_success_schema_standard(self):
        content = (
            'While working on the task: "Determine if unknown substance B is electrically conductive" at workshop,\n'
            "the action 'connect battery anode to orange wire terminal 1' caused 'anode on battery is now connected to terminal 1 on orange wire'.\n"
            'Inventory: {...}\n'
            'Location info: {...}\n'
            "Recent actions: ['a','b']\n"
            "Recent obs: ['x','y']\n"
            'This resulted in a reward: 5, updating the score: 0 -> 5.\n'
            'Marked as SUCCESS.'
        )
        parsed = parse_episode(content)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['room'], 'workshop')
        self.assertIn('connect battery anode', parsed['action'])
        self.assertIn('anode on battery is now connected', parsed['observation'])
        self.assertEqual(parsed['reward_delta'], 5.0)
        self.assertIn('electrically conductive', parsed['task'])

    def test_variability_newlines_and_spaces(self):
        content = (
            'While working on the task:  "Grow plant A"   at   greenhouse,\n\n'
            "the action 'water plant' caused 'The soil looks moist now.'.  \n"
            'This resulted in a reward: +1, updating the score: 3 -> 4.\n'
            'Marked as SUCCESS.'
        )
        parsed = parse_episode(content)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['room'], 'greenhouse')
        self.assertEqual(parsed['reward_delta'], 1.0)
        self.assertEqual(parsed['action'], 'water plant')

    def test_missing_fields_graceful(self):
        # No reward present
        content = (
            'While working on the task: "Boil water" at kitchen,\n'
            "the action 'turn on stove' caused 'The stove is now on.'\n"
        )
        parsed = parse_episode(content)
        self.assertIsNone(parsed)


if __name__ == '__main__':
    unittest.main()


