import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 欠損値の埋め方毎にリスト化
fillna_Minus1_cols = [
    "（紹介予定）入社後の雇用形態",
    "勤務地　最寄駅1（分）",
    "勤務地　最寄駅2（駅からの交通手段）",
    "勤務地　最寄駅1（駅からの交通手段）",
]
fillna_Mean_cols = ["（派遣先）配属先部署　男女比　男", "（派遣先）配属先部署　人数", "（派遣先）配属先部署　平均年齢"]
drop_temp_cols = [
    "勤務地　最寄駅2（駅名）"
    #   "給与/交通費　備考", "（紹介予定）休日休暇", "勤務地　最寄駅2（沿線名）", "勤務地　最寄駅1（沿線名）", "勤務地　最寄駅1（駅名）",
    # "（派遣先）配属先部署", "（紹介予定）年収・給与例", "期間・時間　勤務時間", "仕事内容", "応募資格", "お仕事のポイント（仕事PR）"
]
drop_cols = [
    "お仕事No.",
    "休日休暇　備考",
    "（派遣先）配属先部署　男女比　女",
    "掲載期間　終了日",
    "勤務地　最寄駅2（分）",
    "期間･時間　備考",
    "派遣会社のうれしい特典"
    #  "勤務地　備考", "期間・時間　勤務開始日", "（派遣先）職場の雰囲気", "お仕事名",
]


# データの前処理(欠損値埋め，特徴量の生成)
def preprocessing(train):
    # 欠損値を-1埋め
    for col in fillna_Minus1_cols:
        train[col] = train[col].fillna(-1)

    # 欠損値を平均埋め
    for col in fillna_Mean_cols:
        train[col] = train[col].fillna(train[col].mean())

    # 特徴量の削除
    train = train.drop(drop_temp_cols, axis=1)
    train = train.drop(drop_cols, axis=1)

    # ----- （紹介予定）雇用形態備考
    def converter1(d):
        # （紹介予定）雇用形態備考用の変換器
        if d == "正社員":
            return 1
        elif d == "派遣社員":
            return 0
        else:
            return -1

    train["（紹介予定）雇用形態備考"] = train["（紹介予定）雇用形態備考"].map(converter1)
    # -----

    # ----- 掲載期間　開始日
    # 2019/9/25 -> 1
    # 2019/11/27 -> 0
    train["掲載期間　開始日"] = train["掲載期間　開始日"].map(lambda d: 1 if d == "2019/9/25" else 0)
    # -----

    # ----- （派遣先）概要　勤務先名（漢字）
    # データのあるところ1, データのないところ0
    train["（派遣先）概要　勤務先名（漢字）"] = train["（派遣先）概要　勤務先名（漢字）"].fillna(0)
    train.loc[train["（派遣先）概要　勤務先名（漢字）"] != 0, "（派遣先）概要　勤務先名（漢字）"] = 1
    train["（派遣先）概要　勤務先名（漢字）"] = train["（派遣先）概要　勤務先名（漢字）"].astype(np.int8)
    # -----

    # ----- 拠点番号
    # 前3文字
    train["拠点番号"] = train["拠点番号"].map(lambda d: int(d[:3]))
    # -----

    # ----- （紹介予定）入社時期
    # ※ご紹介先により異なります。詳細はお問い合わせ下さい。 -> -1
    # nan -> -1
    # その他2文字目の値
    train["（紹介予定）入社時期"] = train["（紹介予定）入社時期"].fillna("99")
    train.loc[train["（紹介予定）入社時期"] == "※ご紹介先により異なります。詳細はお問い合わせ下さい。", "（紹介予定）入社時期"] = "99"
    train["（紹介予定）入社時期"] = train["（紹介予定）入社時期"].map(lambda d: int(d[1]))
    train.loc[train["（紹介予定）入社時期"] == 99, "（紹介予定）入社時期"] = -1
    # -----

    # ----- （派遣先）勤務先写真ファイル名
    # 値あり -> 1
    # 値なし -> 0
    train.loc[np.logical_not(train["（派遣先）勤務先写真ファイル名"].isna()), "（派遣先）勤務先写真ファイル名"] = 1
    train["（派遣先）勤務先写真ファイル名"] = train["（派遣先）勤務先写真ファイル名"].fillna(0)
    # -----

    # ----- 勤務地　都道府県コード
    # 東京都のみ市区町村コードに置き換える
    train.loc[train["勤務地　都道府県コード"] == 13, "勤務地　都道府県コード"] = train.loc[
        train["勤務地　都道府県コード"] == 13, "勤務地　市区町村コード"
    ]
    # -----

    # ----- （紹介予定）待遇・福利厚生
    # 値あり -> 1
    # 値なし -> 0
    train.loc[np.logical_not(train["（紹介予定）待遇・福利厚生"].isna()), "（紹介予定）待遇・福利厚生"] = 1
    train["（紹介予定）待遇・福利厚生"] = train["（紹介予定）待遇・福利厚生"].fillna(0)
    # -----

    # ----- "給与/交通費　備考"
    # 値なし -> -1
    # 値あり -> 月収xx万yyyy円 (xx*10000 + yyyy)
    def converter2(d):
        exceptions = {
            "【交通費】◆時給１７６０円の時は日額５００円まで交通費有／時給１８２０円の時は交通費無。どちらか選択できます。",
            "【交通費】◆条件により交通費支給あり。詳細はお気軽にお問合せください。",
        }
        if d in exceptions:
            return -1

        temp1 = int(d.split("】")[1].split("万")[0])
        temp2 = d.split("万")[1].split("円")[0]
        if temp2 == "":
            return temp1 * 10000
        else:
            return temp1 * 10000 + int(temp2)

    train.loc[np.logical_not(train["給与/交通費　備考"].isna()), "給与/交通費　備考"] = train.loc[
        np.logical_not(train["給与/交通費　備考"].isna()), "給与/交通費　備考"
    ].map(converter2)
    train["給与/交通費　備考"] = train["給与/交通費　備考"].fillna(-1)
    # -----

    # ----- "（紹介予定）休日休暇"
    # 値なし -1
    # 値あり 年間休日数
    train.loc[np.logical_not(train["（紹介予定）休日休暇"].isna()), "（紹介予定）休日休暇"] = train.loc[
        np.logical_not(train["（紹介予定）休日休暇"].isna()), "（紹介予定）休日休暇"
    ].map(lambda d: int(d.split("日")[1]))
    train["（紹介予定）休日休暇"] = train["（紹介予定）休日休暇"].fillna(-1)
    # -----

    # ----- 給与/交通費　給与上限
    # 値なし 給与下限の値で埋める
    train.loc[train["給与/交通費　給与上限"].isna(), "給与/交通費　給与上限"] = train.loc[
        train["給与/交通費　給与上限"].isna(), "給与/交通費　給与下限"
    ]
    # -----

    # ----- 勤務地　最寄駅1（沿線名）, 勤務地　最寄駅1（沿線名）
    # labelencording
    le = LabelEncoder()
    train["勤務地　最寄駅1（沿線名）"] = le.fit_transform(train["勤務地　最寄駅1（沿線名）"])
    train["勤務地　最寄駅1（駅名）"] = le.fit_transform(train["勤務地　最寄駅1（駅名）"])
    # -----

    # ----- 勤務地　最寄駅2（沿線名）
    # 値あり -> 1
    # 値なし -> 0
    train.loc[np.logical_not(train["勤務地　最寄駅2（沿線名）"].isna()), "勤務地　最寄駅2（沿線名）"] = 1
    train["勤務地　最寄駅2（沿線名）"] = train["勤務地　最寄駅2（沿線名）"].fillna(0)
    # -----

    # ----- "（紹介予定）年収・給与例"
    train["（紹介予定）年収・給与例"] = train["（紹介予定）年収・給与例"].fillna(-1)
    # income_list = []
    for i in range(len(train["（紹介予定）年収・給与例"])):
        if train["（紹介予定）年収・給与例"][i] != -1:
            if "年収" in train["（紹介予定）年収・給与例"][i]:
                if "年収約" in train["（紹介予定）年収・給与例"][i]:
                    index = train["（紹介予定）年収・給与例"][i].index("年収約")
                    income = train["（紹介予定）年収・給与例"][i][index + 3 : index + 6]
                    if str.isdecimal(income) == 0:
                        if "等" in income:
                            index = train["（紹介予定）年収・給与例"][i].index("万円")
                            income = train["（紹介予定）年収・給与例"][i][index - 3 : index]
                        income = re.sub("\\D", "", income)

                    temp = train["（紹介予定）年収・給与例"]
                    temp.iloc[i] = int(income)
                    train["（紹介予定）年収・給与例"] = temp

                else:
                    index = train["（紹介予定）年収・給与例"][i].index("年収")
                    income = train["（紹介予定）年収・給与例"][i][index + 2 : index + 5]
                    if str.isdecimal(income) == 0:
                        if "等" in income:
                            index = train["（紹介予定）年収・給与例"][i].index("万円")
                            income = train["（紹介予定）年収・給与例"][i][index - 3 : index]
                        income = re.sub("\\D", "", income)

                    temp = train["（紹介予定）年収・給与例"]
                    temp.iloc[i] = int(income)
                    train["（紹介予定）年収・給与例"] = temp

    train["（紹介予定）年収・給与例"] = train["（紹介予定）年収・給与例"].astype(np.int16)
    # -----

    # ----- "（派遣先）配属先部署　平均年齢"
    # 5刻みで年齢層別に分ける -> 5: 5~9, 10: 10~14, 15: 15~19, ...
    train["（派遣先）配属先部署　平均年齢"] = pd.cut(
        train["（派遣先）配属先部署　平均年齢"],
        bins=np.arange(5, 76, 5),
        right=False,
        labels=[5 * i for i in range(1, 15)],
    )
    # -----

    # ----- 期間・時間　勤務時間
    start_list = [0] * len(train["期間・時間　勤務時間"])
    end_list = [0] * len(train["期間・時間　勤務時間"])
    working_hours_list = [0] * len(train["期間・時間　勤務時間"])
    for i in range(len(train["期間・時間　勤務時間"])):
        start_index = train["期間・時間　勤務時間"][i].index(":", 0, 3)
        end_index = train["期間・時間　勤務時間"][i].index(":", 5, 10)
        start_time = (
            int(train["期間・時間　勤務時間"][i][:start_index])
            + int(train["期間・時間　勤務時間"][i][start_index + 1 : start_index + 3]) / 60
        )
        start_list[i] = start_time
        if str.isdecimal(train["期間・時間　勤務時間"][i][end_index - 2 : end_index]) == 0:
            end_list[i] = 0
            working_hours_list[i] = 0
        else:
            end_time = (
                int(train["期間・時間　勤務時間"][i][end_index - 2 : end_index])
                + int(train["期間・時間　勤務時間"][i][end_index + 1 : end_index + 3]) / 60
            )
            end_list[i] = end_time
            working_hours = end_time - start_time
            working_hours_list[i] = working_hours
    train["start_time"] = start_list
    train["end_time"] = end_list
    train["working_hours"] = working_hours_list

    train = train.drop("期間・時間　勤務時間", axis=1)
    # -----

    # ----- 仕事内容
    # どの業界からの依頼か
    train["仕事内容"] = train["仕事内容"].map(lambda d: "コンサル" if "コンサル" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "不動産" if "不動産" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "機械" if "機械" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "商社" if "商社" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "建設" if "建設" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "システム" if "システム" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "企画" if "企画" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "医薬品" if "医薬品" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "通信" if "通信" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "金融" if "金融" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "事務所" if "事務所" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "証券" if "証券" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "食品" if "食品" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "デザイン" if "デザイ" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "人材サービス" if "人材サービス" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "電気" if "電気" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "飲食" if "飲食" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "システム" if "ソフトウェア" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "物流" if "物流" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "化学" if "化学" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "印刷" if "印刷" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "リース" if "リース" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "ホテル" if "ホテル" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "旅行" if "旅行" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "福祉" if "福祉" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "教育" if "教育" in d else d)
    train["仕事内容"] = train["仕事内容"].map(lambda d: "広告" if "広告" in d else d)
    # -----

    # ----- "お仕事名"
    # 出現頻度で変換
    freq = train["お仕事名"].value_counts()
    train["お仕事名"] = train["お仕事名"].map(freq)
    # -----

    # ----- "お仕事のポイント（仕事PR）"
    freq = train["お仕事のポイント（仕事PR）"].value_counts()
    train["お仕事のポイント（仕事PR）"] = train["お仕事のポイント（仕事PR）"].map(freq)
    # -----

    return train


# ターゲットエンコーディング(OutOfFold)と出現頻度特徴量を追加する関数の定義
cat_cols = [
    "勤務地　最寄駅1（沿線名）",
    "勤務地　最寄駅1（駅名）",
    "勤務地　備考",
    "期間・時間　勤務開始日",
    "（派遣先）配属先部署",
    "（派遣先）職場の雰囲気",
    "仕事内容",
    "拠点番号",
    "応募資格",
]


def targetEncode(test_x, train_x, train_y, cat_cols, kf):

    for c in cat_cols:

        # Minorityデータをまとめる(出現回数が10回未満のものをまとめる)
        freq = train_x[c].value_counts()
        train_x[c] = train_x[c].map(freq)
        train_x.loc[train_x[c] < 10, c] = -99

        # 学習データ全体で各カテゴリにおける給料の平均を計算
        data_tmp = pd.DataFrame({c: train_x[c], "target": train_y})
        target_mean = data_tmp.groupby(c)["target"].mean()
        # テストデータのカテゴリを置換
        test_x[c] = test_x[c].map(target_mean)

        # targetを付加
        data_tmp = pd.DataFrame({c: train_x[c], "target": train_y})
        # 変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train_x.shape[0])

        # 学習データからバリデーションデータを分ける
        for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

            # 学習データについて、各カテゴリにおける目的変数の平均を計算(smoothing無)
            # target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()

            # 学習データについて、各カテゴリにおける目的変数の平均を計算(smoothing有 -> 出現回数が少ないものに対する補正有)
            target_sum = (
                data_tmp.iloc[tr_idx].groupby(c)["target"].sum() + train_y.mean() * 10
            )
            target_count = data_tmp.iloc[tr_idx].groupby(c)["target"].count() + 10
            target_mean = target_sum / target_count

            # バリデーションデータについて、変換後の値を一時配列に格納
            tmp[va_idx] = train_x[c].iloc[va_idx].map(target_mean)

        # 変換後のデータで元の変数を置換
        train_x[c] = tmp

        # train_x = train_x.drop(c+"出現頻度",axis=1)
    return train_x
