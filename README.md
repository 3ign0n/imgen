日本語／英語のプロンプトから、画像を生成するプログラムです

## CLI
###　環境セットアップ
1. 必要なライブラリインストール
```python
pip install -rq requirements.txt
```

2. 学習モデルを、modelsフォルダ配下に配置

### 実行
```python
python imgen.py <プロンプト>
```
※ プロンプト部分には、描画させたい画像を表す言葉を入力します。空白や特殊記号を含む場合は、クオテーションで囲うなり、エスケープするなり、適宜、シェルが要求するルールに従ってください。
※ outputディレクトリに、生成した画像が吐き出されます。
※ モデルが重いので、GPUなしだと、20分ぐらいかかるかもしれません。


### 学習
ディスク容量等の都合で、モデルの学習は、train.ipynbを使ってGoogle Colaboratoryで実行しました。


## 学習データについて
### 画像＆英語のキャプション
https://cocodataset.org/#download
から、
2017 Train/Val annotations
を入手し、captions_train2017.jsonに入っている情報を使います。

18GBもあって使いづらいので、使うデータのみを絞り込みます。

### 日本語キャプション
https://github.com/STAIR-Lab-CIT/STAIR-captions
から、入手したstair_captions_v1.2_train.jsonに入っている情報を使います。

このjsonは使いづらいので、
```shell
cat stair_captions_v1.2_train.json| jq -r '["image_id","caption"],(.annotations[]|[.image_id,.caption])|@csv' > stair_captions_v1.2_train_captions1.csv

# 特定のキーワードを持つ画像のみに絞り込み
grep  -e '猫' -e '犬' -e '鳥' -e '魚' -e '馬' -e '飛行機' -e '船' -e '野菜' stair_captions_v1.2_train_captions1.csv > stair_captions_v1.2_train_captions2.csv
cat stair_captions_v1.2_train.json| jq -r '["image_id","width","height","file_name","url"],(.images[]|[.id,.width,.height,.file_name,.flickr_url])|@csv' > stair_captions_v1.2_train_images.csv
```
とします。

英語も日本語も、1つの画像に対するキャプションが複数あり（最大5個）、学習時間を短縮するため、1つの画像に対するキャプションは英語、日本語、それぞれ1つずつとしました。


## モデルについて
https://www.kaggle.com/code/mennaalaarasslan/conditional-ddpm/notebook
を参考に、conditionalな、Denoising Diffusion Probabilistic Modelです。
conditionは、日本語で学習させたCLIPの改良版であるCLOOBを使いました。
https://huggingface.co/rinna/japanese-cloob-vit-b-16

