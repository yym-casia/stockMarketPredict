#!/bin/bash
set -e
APP_DIR="$HOME/stockMarketPredict"
VENV="$APP_DIR/.venv"
cd "$APP_DIR"

if [ ! -d "$VENV" ]; then
  echo "==> create venv"
  python3 -m venv "$VENV"
fi
echo "==> pip install"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q -r requirements-server.txt

echo "==> update systemd"
sudo tee /etc/systemd/system/stockService.service > /dev/null <<EOF
[Unit]
Description=Stock Market Predict Dashboard
After=network.target

[Service]
ExecStart=$VENV/bin/python $APP_DIR/serve_dashboard.py
WorkingDirectory=$APP_DIR
User=ubuntu
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable stockService
sudo systemctl restart stockService
sleep 3
sudo systemctl status stockService --no-pager | head -18
curl -s -o /dev/null -w "dashboard HTTP %{http_code}\n" http://127.0.0.1:8088/dashboard.html
echo "Deploy OK: http://82.157.98.97:8088/dashboard.html"
