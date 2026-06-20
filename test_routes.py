import sys
sys.path.insert(0, 'c:\\Users\\pc\\Desktop\\bot')

# Mock the WebSocket to prevent connection attempts
import unittest.mock as mock
sys.modules['websocket'] = mock.MagicMock()

# Now import the bot
from pocket_option_bot import app

print("\n=== REGISTERED ROUTES ===")
for rule in app.url_map.iter_rules():
    print(f"{rule.rule:30} {rule.endpoint:20} {rule.methods}")
