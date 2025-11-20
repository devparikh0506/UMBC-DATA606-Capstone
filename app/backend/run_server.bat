@echo off
echo Starting Django ASGI server with WebSocket support...
daphne -b 0.0.0.0 -p 8000 bci_app.asgi:application

