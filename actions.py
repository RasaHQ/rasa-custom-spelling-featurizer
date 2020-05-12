# from typing import Any, Text, Dict, List

# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher


# class ActionProglang(Action):

#     def name(self) -> Text:
#         return "action_proglang"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         print(tracker.latest_message)
#         found_ents = [e['value'] for e in tracker.latest_message['entities']]
#         dispatcher.utter_message(text=f"You want to talk about code! About {''.join(found_ents)} perhaps?")
#         return []
