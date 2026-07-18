# setup.py
from setuptools import setup, find_packages

setup(
    name='xarm_sdk',  # 包名
    version='0.2.0',           # 版本号
    description='XARM机器人SDK', # 包描述
    long_description=open('README.md').read(), # 详细描述，通常来自 README
    long_description_content_type='text/markdown', # 描述类型
    author='XARM',        # 作者
    author_email='your_email@example.com', # 作者邮箱
    url='http://10.0.3.101/EAI-manipulator/xarm_sdk', # 项目主页
    packages=find_packages(), # 自动查找所有包
    # 或者手动指定: packages=['my_package', 'my_package.submodule'],
    # 如果你的包在 src 目录里: package_dir={'': 'src'}, packages=['my_package']
    install_requires=[         # 依赖项
        'requests>=2.20.0',
        'numpy',
    ],
    classifiers=[              # 分类，有助于 PyPI 搜索
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',   # 支持的Python版本
)