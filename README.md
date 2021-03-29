# Flowers-Recognition
## 今回のやったこと
様々なモデルを学習し、Light-gbmでアンサンブル

### 学習させてみたモデル
* Vision Transformer (resnetのハイブリッド版も使用)
* SE-ResNeXt
・SK-ResNeXt
・Efficientnet_b2~b4
・Inception_Resnet_v2

### アンサンブルに使用したモデル
・Vision Transformer (vit_base_patch16_224_ver2_vit_base_patch16_224)  
・SE-ResNeXt (seresnext50_32x4d_seresnext50_32x4d)
・Efficientnet_b2(tf_efficientnet_b2_ver2_tf_efficientnet_b2_oof)

### special thanks
今回は,前回のgalaコンペの村上さんと田中さんのコードにかなり助けて頂きました。ありがとうございます！