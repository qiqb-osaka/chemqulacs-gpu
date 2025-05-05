# 開発者向け注意書き

## pennylaneを用いてテストをする場合の注意
- pennylaneは左から0-index, qulacsは右から0-index.
- 回転角も逆.
よって, pennylaneの値と比較する場合, qulacs側がpennylaneの再現をするようにする. つまり, wiresとstateのindexを通常の意味と逆にする.
例えば,
pennylane: |1100> の時, qulacsでも|1100>とし,
pennylane: wires=[0,1,2,3]の時, qulacsはwires=[3,2,1,0]とする.
すると, pennylaneが行なっている操作と同じ操作を行うことになる.
これは例えばqubits=4でpennylaneがCNOT(0,1)をしている時, qulacsはCNOT(2,3)を行なっていることを考えると分かりやすい. |1100>とすることも, pennylaneの0-indexに合わせて電子を置いていると考えればよい.