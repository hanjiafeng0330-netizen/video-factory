"""部署平台识别的 ASGI 入口。"""

from app.bootstrap.dev import create_dev_app

app = create_dev_app()
