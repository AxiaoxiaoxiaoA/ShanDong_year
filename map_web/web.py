from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# 主页
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"<h1>错误</h1><p>{str(e)}</p><p>请确保 templates/index.html 文件存在</p>"

# 提供静态文件
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# 其他页面
@app.route('/about')
def about():
    return render_template('about.html')

# 测试页面
@app.route('/test')
def test():
    return render_template('test.html')

if __name__ == '__main__':
    # 获取本机 IP
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    port = 8888
    
    print('=' * 50)
    print(f'服务器已启动!')
    print(f'本机访问: http://localhost:{port}')
    print(f'本机访问: http://127.0.0.1:{port}')
    print(f'局域网访问: http://{local_ip}:{port}')
    print('=' * 50)
    
    # 运行服务器（绑定到本机 IP 支持局域网访问）
    try:
        app.run(host=local_ip, port=port, debug=False)
    except OSError as e:
        print(f'启动失败: {e}')
        print('尝试使用 localhost...')
        app.run(host='localhost', port=port, debug=False)