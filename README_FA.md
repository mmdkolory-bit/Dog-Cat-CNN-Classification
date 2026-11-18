# پروژه طبقه‌بندی سگ و گربه با CNN

**درس/موضوع:** یادگیری عمیق و پردازش تصویر  
**نوع مسئله:** طبقه‌بندی دودویی تصاویر  
**مدل:** شبکه عصبی کانولوشنی (CNN) آموزش‌دیده از ابتدا با PyTorch

این بسته شامل کد آموزش، ارزیابی، پیش‌بینی یک تصویر جدید، آزمایش اثر عمق شبکه و لایه‌های منظم‌سازی، نمودارهای یادگیری و تولید خودکار گزارش است.

## ۱. دریافت دیتاست

دیتاست مورد استفاده:

https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

پس از دانلود و استخراج، ساختار معمول آن به این شکل است:

```text
PetImages/
├── Cat/
│   ├── 0.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    └── ...
```

خود تصاویر به دلیل حجم و شرایط مجوز در این بسته قرار نگرفته‌اند. برنامه پوشه‌های `Cat` و `Dog` را به‌صورت بازگشتی جست‌وجو می‌کند و تصاویر خراب/ناخوانا را کنار می‌گذارد.

### روش راحت‌تر در Google Colab

در یک سلول Colab اجرا کنید:

```python
!pip -q install kagglehub
import kagglehub
DATA_DIR = kagglehub.dataset_download(
    "bhavikjikadara/dog-and-cat-classification-dataset"
)
print(DATA_DIR)
```

در صورتی که Kaggle برای دریافت مجموعه احراز هویت بخواهد، ابتدا مجوز/توکن Kaggle خود را در محیط Colab تنظیم کنید یا فایل دیتاست را دستی دانلود و استخراج کنید.

## ۲. نصب وابستگی‌ها

Python 3.10 یا 3.11 پیشنهاد می‌شود. در محیط مجازی:

```bash
python -m venv .venv
```

فعال‌سازی در Windows:

```bat
.venv\Scripts\activate
```

فعال‌سازی در Linux/macOS:

```bash
source .venv/bin/activate
```

سپس:

```bash
pip install -r requirements.txt
```

برای نصب PyTorch متناسب با کارت گرافیک، دستور رسمی مناسب سیستم‌عامل و نسخه CUDA را از صفحه انتخاب‌گر نصب PyTorch بگیرید: https://pytorch.org/get-started/locally/ . اگر CUDA در دسترس نباشد، کد به‌صورت خودکار روی CPU اجرا می‌شود.

## ۳. آموزش مدل اصلی

از ریشه پروژه:

```bash
python train.py --data-dir "مسیر/به/PetImages" --epochs 12 --batch-size 16 --image-size 128
```

نمونه در ویندوز:

```bat
python train.py --data-dir "D:\Datasets\PetImages" --epochs 12 --batch-size 16 --image-size 128
```

برای حافظه گرافیکی محدود، `--batch-size 16` مقدار پیش‌فرض است؛ برای حافظه گرافیکی کمتر از ۴ گیگابایت، مقدار `8` را امتحان کنید. در سیستم‌های CPU ضعیف می‌توان برای آزمایش اولیه تعداد epoch را ۲ یا ۳ قرار داد؛ برای گزارش نهایی از آموزش کامل‌تر استفاده کنید.

## ۴. آزمایش اثر لایه‌ها و عمق شبکه

برای اجرای مدل اصلی و سپس آزمایش‌های مقایسه‌ای در یک مرحله:

```bash
python train.py --data-dir "مسیر/به/PetImages" --epochs 12 --batch-size 16 --image-size 128 --run-ablation
```

اگر مدل اصلی را قبلاً اجرا کرده‌اید و فقط آزمایش‌های مقایسه‌ای باقی مانده است، از این گزینه استفاده کنید تا مدل پایه دوباره آموزش داده نشود:

```bash
python train.py --data-dir "مسیر/به/PetImages" --epochs 12 --batch-size 16 --image-size 128 --ablation-only
```

آزمایش‌ها:

| نام آزمایش | بلوک‌های کانولوشنی | Batch Normalization | Dropout | هدف |
|---|---:|---|---|---|
| CNN_3Blocks_Baseline | 3 | دارد | دارد | مدل مرجع |
| CNN_2Blocks | 2 | دارد | دارد | اثر کاهش عمق |
| CNN_4Blocks | 4 | دارد | دارد | اثر افزایش عمق |
| CNN_NoBatchNorm | 3 | ندارد | دارد | اثر حذف Batch Normalization |
| CNN_NoDropout | 3 | دارد | ندارد | اثر حذف Dropout |

در همه آزمایش‌ها تقسیم داده و تنظیمات آموزش ثابت نگه داشته می‌شود. انتخاب بهترین وزن هر آزمایش بر اساس Validation Loss انجام می‌شود و سپس مدل روی Test ارزیابی می‌شود.

## ۵. معیارها و تعریف آن‌ها

ترتیب برچسب‌ها `0 = Cat` و `1 = Dog` است. در محاسبه Precision، Recall و F1، کلاس مثبت **Dog** در نظر گرفته شده است.

- **Accuracy:** نسبت پیش‌بینی‌های درست به کل نمونه‌ها.
- **Precision:** از مواردی که سگ پیش‌بینی شده‌اند، چه نسبتی واقعاً سگ بوده‌اند.
- **Recall:** از تمام تصاویر واقعی سگ، چه نسبتی پیدا شده‌اند.
- **F1-score:** میانگین هارمونیک Precision و Recall.
- **Confusion Matrix:** سطرها برچسب واقعی و ستون‌ها پیش‌بینی‌اند؛ ترتیب هر دو `Cat, Dog` است.
- **MSE:** در این پروژه میانگین مربع اختلاف احتمال پیش‌بینی کلاس سگ و برچسب صفر/یک است؛ این معیار به نام Brier score نیز شناخته می‌شود. تابع هزینه آموزش Cross-Entropy است و با MSE تفاوت دارد.

فرمول MSE:

```text
MSE = (1/N) * sum((P(Dog | x_i) - y_i)^2)
```

که در آن `y_i=0` برای گربه و `y_i=1` برای سگ است.

## ۶. خروجی‌های پروژه

پس از اجرا، پوشه `outputs/` ساخته می‌شود:

```text
outputs/
├── data_splits.csv
├── dataset_summary.json
├── REPORT.md
├── main/
│   ├── best_model.pth
│   ├── history.csv
│   ├── test_metrics.json
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── mse_epochs.png
│   ├── learning_curves.png
│   ├── sample_predictions.png
│   └── test_predictions.csv
└── ablations/
    ├── experiment_summary.csv
    └── architecture_comparison.png
```

`REPORT.md` بعد از اجرا به‌طور خودکار با مقادیر واقعی مجموعه آزمون و جدول آزمایش‌ها تکمیل می‌شود.

## ۷. پیش‌بینی یک تصویر جدید

بعد از آموزش:

```bash
python predict.py "مسیر/تصویر.jpg"
```

یا مسیر checkpoint را مشخص کنید:

```bash
python predict.py "cat.jpg" --checkpoint outputs/main/best_model.pth --image-size 128
```

## ۸. نکات علمی برای تحلیل نتایج

1. صرفاً Accuracy را گزارش نکنید؛ Precision، Recall، F1 و ماتریس ابهام را نیز بررسی کنید.
2. نمودار MSE آموزش و اعتبارسنجی را مقایسه کنید. کاهش MSE آموزش همراه با افزایش MSE اعتبارسنجی می‌تواند نشانه بیش‌برازش باشد.
3. تعداد بلوک بیشتر الزاماً عملکرد بهتر ایجاد نمی‌کند؛ پارامتر بیشتر ممکن است هزینه محاسباتی را زیاد کند یا در صورت کافی نبودن داده/تنظیم‌سازی باعث بیش‌برازش شود.
4. اثر Batch Normalization و Dropout را از جدول واقعی `experiment_summary.csv` نتیجه‌گیری کنید، نه بر اساس انتظار نظری.
5. مجموعه آزمون در انتخاب وزن‌ها استفاده نشده است؛ معیارهای آزمون پس از انتخاب checkpoint بر اساس Validation Loss محاسبه می‌شوند.

## ۹. محدودیت و بازتولیدپذیری

- Seed پیش‌فرض ۴۲ است و فایل `data_splits.csv` مسیر و برچسب همه تصاویر و نوع split را ذخیره می‌کند.
- نتیجه دقیق به نسخه کتابخانه‌ها، سخت‌افزار و اجرای آموزش بستگی دارد؛ بنابراین اعداد عملکرد از پیش در این بسته درج نشده‌اند.
- برای اجرای کامل آزمایش Ablation زمان بیشتری نسبت به آموزش یک مدل لازم است.

## منبع داده

Bhavik Jikadara, *Dog and Cat Classification Dataset*, Kaggle.  
https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
