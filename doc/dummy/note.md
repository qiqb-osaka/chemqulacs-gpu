# note
このフォルダはchemqulacs_cppドキュメント作成用のモックになっている  

shpinxの仕様でバイナリファイルとstubは同名でなければならないのでこうなっている。  

またchemqulacs_cpp.pyiは__init__.pyiのシンボリックリンクなのでchemqulacs_cppのstubを更新する際は必ず__init__.pyiの方を更新すること。  

## ダミーのchemqulacs_cppの作り方
以下のコマンドを実行する
```bash
cd ./doc/dummy
cp chemqulacs_cpp.pyi chemqulacs_cpp.pyx
CC=gcc poetry run python setup.py build_ext --inplace
```
