- # LDA实践中存在的一些疑问

- 数据集中包含了德语和英语，主要为德语（约100多条），所以为了方便对文本做预处理先使用了langdetect对语料分类

- 之后单独对德语进行一次主题建模，对德语英语混合进行了一次主题建模，但效果都是困惑度和一致性分数随着主题分类数量上升而上升（主题数范围2~99），不太清楚什么原因

- 可能需要对德语语料进行特殊的分词操作

- 详细得到的结果在main.ipnb中，可视化页面分别在result（记录了主题数量0-50）和result_de（记录了主题数量0-50）中

- 以下是关于主题数量、一致性分数和困惑度的曲线图

  - 混合语料

    ![image-20240704221058677](image\image-20240704221058677.png)

  - 德语语料(主题数量范围在0-50之间时变化趋势和混合语料基本一致，以下为50-100之间)

    ![image-20240704221221021](image\image-20240704221221021.png)

  

  