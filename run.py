import os
import io

from flask import (
    Flask,
    flash,
    request,
    redirect,
    render_template,
    make_response,
)
from werkzeug.utils import secure_filename
from sklearn.model_selection import KFold
import joblib
import pandas as pd

from preprocessing_csv import preprocessing, targetEncode, cat_cols

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = b"zK39MwLNpEI884VATmu708UhhvvTaSmw"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


test_csv = pd.DataFrame(index=range(1))


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":

        # fileでpostされていない場合メッセージを出す
        if "file" not in request.files:
            flash("指定タグのファイルをpostしてください")
            return redirect(request.url)

        file = request.files["file"]
        # ファイル未添付でuploadした場合メッセージを出す
        if file.filename == "":
            flash("ファイルが選択されていません")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)

            # 受信ファイルをcsvファイルとして読み込み
            uni_string = file.stream.read()
            test_or_x = pd.read_csv(io.BytesIO(uni_string), encoding="utf8")

            train_x = pd.read_csv("train_x.csv")
            train_y = pd.read_csv("train_y.csv")
            # test_or_x = pd.read_csv("test_x.csv")

            # データの中で全て1のものや，記入のないものをdropする
            train_x = train_x.dropna(how="all", axis="columns")
            for col in train_x.columns:
                if len(train_x[col].value_counts()) == 1:
                    train_x = train_x.drop(col, axis=1)

            # 前処理
            train_x = preprocessing(train_x)
            test_x = preprocessing(test_or_x)

            # ターゲットエンコーディング実行
            targetEncode(
                test_x,
                train_x,
                train_y["応募数 合計"],
                cat_cols,
                KFold(n_splits=10, shuffle=True, random_state=42),
            )

            test_x = test_x[train_x.columns]

            # pkl = sklearn.externals.joblib.load("predictor_dt.pkl")
            pkl = joblib.load("predictor_dt.pkl")

            # モデルの実行
            predictor = pkl.predict(test_x)

            predict_pd = pd.Series(predictor)
            submit = pd.concat([test_or_x["お仕事No."], predict_pd], axis=1)
            submit.columns = ["お仕事No.", "応募数 合計"]

            global test_csv
            test_csv = submit

            return render_template("download.html")
    return render_template("upload.html")


@app.route("/download", methods=["GET", "POST"])
def download_file():
    global test_csv

    if test_csv.size != 0:
        resp = make_response(test_csv.to_csv())
        resp.headers["Content-Disposition"] = "attachment; filename=test_submit.csv"
        resp.headers["Content-Type"] = "text/csv"
        test_csv = pd.DataFrame(index=range(1))
        return resp
    else:
        flash("ダウンロード対象のファイルがありません")
        return render_template("upload.html")


if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=50000)
