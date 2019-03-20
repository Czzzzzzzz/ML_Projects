# 建模Quiz

## 题目

简单的二分类任务

## 要求

* 模型要求：无，lr、svm、tree、dnn均可
* 语言要求：python

## 数据说明

* train.data：5000条
* test.data：3000条

数据包含两列：feature、target其中

* feature：已经过一定的加工和脱敏，key:value形式，key是特征编号，value是特征值
* target：分类label

## 评价指标

* AUC：不多做解释
* KS：一种评价模型区分能力的指标

以上两个指标可以使用

> pip install zzhfun

安装zzhfun模块，之后使用

```python
auc, ks = cal_auc_ks(y_true, y_pred)
```
计算得到，关于模块和函数的说明如有需要请参考[github](https://github.com/FlashSnail/zzhfun)

**后续我会使用跨时间测试集验证模型性能，请大家注意过拟合问题**


## 交付文件

请打包以下文件：

* 建模代码
* 模型文件
* 测试数据
* 测试代码（可python test.py 直接执行）
* report表格

| 使用的算法 | 建模思路过程概述 | AUC | KS |
| :--------: | :--------: | :--------: | :--------: |
| Lightgbm | 1. 数据分析：a. 统计缺失值。Train中缺失值超过98%的共1499维。b. 数据均衡。c. |       |       |

以上report表格需要在邮件正文中贴出

## 特别说明

**数据已经经过加工与脱敏，切勿传播，如有传播，保留追求法律责任的权利，请大家遵守数据安全法规**