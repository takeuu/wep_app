import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# タイトル
st.title('どこかおかしいけどお試しvar')

# ファイルアップロード
uploaded_file = st.file_uploader('', type=['csv'])

if uploaded_file is not None:
    # データの読み込み
    data = pd.read_csv(uploaded_file)
    st.write('アップロードされたデータ:')
    st.write(data.head())

    # ターゲット変数と説明変数の入力
    target = st.selectbox('ターゲット変数を選択してください', data.columns)
    features = st.multiselect('説明変数を選択してください', data.columns)

    if st.button('因果推論を実行'):
        if target and features:
            try:
                # ターゲット変数の二値化（中央値を閾値とする例）
                threshold = data[target].median()
                data[target] = data[target].apply(lambda x: 1 if x > threshold else 0)

                # 説明変数とターゲット変数をデータフレーム形式で取得
                X = data[features]
                y = data[target]

                # ロジスティック回帰モデルを使用して傾向スコアを計算
                model = LogisticRegression()
                model.fit(X, y)
                data['propensity_score'] = model.predict_proba(X)[:, 1]

                # 傾向スコアに基づくマッチング
                treatment = data[data[target] == 1]
                control = data[data[target] == 0]

                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(control[['propensity_score']])
                distances, indices = nn.kneighbors(treatment[['propensity_score']])
                matched_control = control.iloc[indices.flatten()]

                # マッチング後のデータセット
                matched_data = pd.concat([treatment, matched_control])

                # 傾向スコアの分布をヒストグラムで表示
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(treatment['propensity_score'], bins=20, alpha=0.5, label='Treatment')
                ax.hist(control['propensity_score'], bins=20, alpha=0.5, label='Control')
                ax.set_xlabel('Propensity Score')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.set_title('Distribution of Propensity Scores')

                st.pyplot(fig)

                st.write('因果推論の結果:')
                st.write(matched_data.head())
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
        else:
            st.error('ターゲット変数と説明変数を選択してください。')
