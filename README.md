# MuJoCo Robot Arm Control (FetchReach)

## 概要
MuJoCo + Gymnasium Robotics を用いて、
ロボットアームの制御および模倣学習の基礎実装を行ったプロジェクト。

## 内容
- ランダム制御
- 目標位置への単純制御
- データ収集（imitation learning用）

## 環境構築
```bash
pip install -r requirements.txt
```

## 実行方法

### ランダム動作
```bash
python run_random.py
```

### 制御付き動作
```bash
python run_controller.py
```

### データ収集
```bash
python collect_data.py
```

## 技術スタック
- MuJoCo
- Gymnasium Robotics
- Python

## 今後の拡張
- imitation learningによる学習
- 強化学習の導入
