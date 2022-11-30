from setuptools import setup

setup(name='q_squared',
      version='1.04',
      description='A reference-free metric that aims to evaluate the factual consistency of knowledge-grounded dialogue systems',
      url='https://github.com/nickseek/q-squared/tree/split_long',
      author='Or Honovich',
      author_email='or.honovich@gmail.com',
      license='Apache License 2.0',  
      packages=['q_squared',],
      install_requires=[
          'numpy', 'pandas','bert-score','spacy','torch','transformers','sentencepiece', 'protobuf==3.20'
      ],
      include_package_data=True,
      zip_safe=False
)



