import os

from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='Memoir',
        version='1.0.0',
        description='Package to colorize black and white videos.',
        # scripts=['bin/'],
        url='https://github.com/andy6975/Memoir/',
        author='Shivam',
        author_email='vs.251618@gmail.com',
        packages=['memoir', 'memoir.data'],
        license='MIT',
        long_description=open('README.md').read(),
        zip_safe=False,
        install_requires=requirements,
        keywords='deep_learning machine_learning tensorflow video_color black&white',
        classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: College_students',
        'Intended Audience :: Hobby/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Ubuntu',
        'Programming Language :: Python :: 3.6',
        'Topic :: Research',
        'Topic :: Research :: Video Colorization',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ])
