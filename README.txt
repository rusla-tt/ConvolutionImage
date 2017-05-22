[構成]
app/app.py WebAPI的なあれ
app/lib/ WebAPIから呼び出される色んなもの
configure.py
configディレクトリにあるYamlファイルを読み込んでロードする為だけに
存在するファイル。特に説明事項なし。
ImagePkl.py
学習用イメージファイルを加工してPickleファイルに変換する処理が
記述されたファイル。Pickleファイルは色々なものが詰め込めるバイ
ナリファイルです。機械学習で使われるものではないのですが、色々
便利なのでPickle化してます。
KerasNeauralNetwork.py
機械学習用の本体Pythonファイル。利用ライブラリはKerasを用いている。
バックグラウンドではTensorflowが動作しています。
ConvolutionNeauralNetworkという手法を用いて今回は学習及び判別を
行なっています。
内部を話すとまたややこしいのですが、入出力層を合わせると5層構造
をとるニューラルネットワークを構築しています。
SonnetNeauralNetwork.py(未完成未ロード)
KerasをSonnetで書き直したもの。こちらもGoogle製
Tensorflowが裏で動くが、Tensorflow + Sonnetなどの書き方が可能なので
Kerasより細かい設定を行うことが可能です。
しかし、ものすごく難しい。時間足りず実装が中途半端。

data/
学習データの元データとPickleファイルが配置されるディレクトリ
今回の学習データの一部をみたい時などはこのディレクトリを参照のこと

models/
学習した結果を持っておくディレクトリ
バイナリファイルなのでみてもよくわからないと思います。
jsonファイルは機械学習する時に使う重みの設定などが格納されています。
ただ、このjsonファイルを見ただけではよくわからないので説明も割愛。

tmp/
Flaskで受け取ったImageファイルを格納するためのディレクトリ。
特に何かを想定してる訳ではないのでtmp.jpgとして保存される

[使い方]
READMEがおいてある場所をPYTHON_APP_HOMEとした場合、その配下で
python app/app.py
とするとFlaskが起動してAPIとして利用できる状態になります。
あとはモデルが作りたければ
curl http://ipaddress:5000/model -X CREATE
pklが作りたい場合は
curl http://ipaddress:500/pkl -X CREATE
などやってもらえればOKかなと思います。



