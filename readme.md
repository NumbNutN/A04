### 代码规范

+ 供其他人使用的代码最好写好功用和参数、返回值等注释，参数和返回值类型最好显式的标注数据类型
+ 注释推荐采用VSCode插件"autoDocstring - Python Docstring Generator"进行模板生成

### 项目组织规范

+ 可以被共享的工具类方法可以放在git主目录/tool下，作为tool工具包的一部分
+ 其余代码可以存放在其他区域

### 提交规范

+ 可以使用 pull request，也可以直接push
+ 分支不要求干净

### 用例介绍

+ classification_base_on_BOW.py

一个基于词袋模型特征模型和简单线性分类器的分类模型的Hello World
采集了train2.csv 婚恋交友 和 信贷理财 两个标签的 线性分类模型

+ photo_gets.py

selenium实现对网页快照的截取(图片形式)

### 项目依赖

python解释器 3.7.16
  
spacy 3.5.0
+ tool/classification_tool.py
+ tool/feature_extraction_tool.py
+ classification_base_on_BOW.py

numpy 1.21.6
+ tool/evaluation_tool.py
+ classification_base_on_BOW.py
  
selenium 4.8.2
+ photo_gets.py
  
scikit-learn 1.0.2
+ classification_base_on_BOW.py
+ tool/feature_extraction_tool.py

openpyxl 3.1.2
+ tool/classification_tool.py

pillow 9.4.0
+ photo_gets.py

pytesseract 0.3.10
+ photo_gets.py

#### 模型
+ zh_core_web_md
```
nlp = spacy.load("zh_core_web_md)
```
+ zh_core_web_sm
```
nlp = spacy.load("zh_core_web_sm)
```