from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _pct(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def generate_report(output_dir: str | Path, dataset_counts: dict, split_counts: dict,
                    image_size: int = 128, epochs: int = 12,
                    learning_rate: float = 1e-3, batch_size: int = 16, seed: int = 42):
    output_dir = Path(output_dir)
    metrics_path = output_dir / "main" / "test_metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    matrix = metrics["confusion_matrix"]
    ablation_path = output_dir / "ablations" / "experiment_summary.csv"
    ablation_table = "آزمایش مقایسه‌ای اجرا نشده است. برای اجرای آن دستور `--run-ablation` را اضافه کنید."
    if ablation_path.exists():
        df = pd.read_csv(ablation_path)
        cols = ["experiment", "blocks", "batchnorm", "dropout", "parameter_count",
                "training_seconds", "best_epoch", "best_val_accuracy", "best_val_loss",
                "test_accuracy", "test_precision", "test_recall", "test_f1", "test_mse"]
        present = [col for col in cols if col in df.columns]
        df = df[present].copy()
        for col in ["best_val_accuracy", "best_val_loss", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_mse"]:
            if col in df:
                df[col] = df[col].map(lambda x: f"{x:.4f}")
        if "training_seconds" in df:
            df["training_seconds"] = df["training_seconds"].map(lambda x: f"{x:.1f}s")
        if "parameter_count" in df:
            df["parameter_count"] = df["parameter_count"].map(lambda x: f"{int(x):,}")
        ablation_table = df.to_markdown(index=False)

    report = f'''# گزارش پروژه: طبقه‌بندی تصاویر گربه و سگ با شبکه عصبی کانولوشنی (CNN)

## ۱. هدف پروژه
هدف، طراحی و آموزش یک شبکه عصبی کانولوشنی از ابتدا برای تشخیص دو کلاس «گربه» و «سگ» و سنجش تعمیم‌پذیری آن روی مجموعه آزمون مستقل است.

## ۲. داده‌ها و آماده‌سازی
- تعداد تصاویر معتبر شناسایی‌شده: **{dataset_counts['total']:,}**
- گربه: **{dataset_counts['cat']:,}** | سگ: **{dataset_counts['dog']:,}**
- تقسیم‌بندی طبقه‌بندی‌شده: آموزش **{split_counts['train']:,}**، اعتبارسنجی **{split_counts['validation']:,}**، آزمون **{split_counts['test']:,}** تصویر.
- تصاویر به RGB تبدیل و به اندازه ثابت **{image_size}×{image_size}** تغییر اندازه داده شده‌اند.
- افزایش داده فقط برای آموزش شامل وارونگی افقی، چرخش محدود و تغییر جزئی روشنایی/کنتراست/اشباع است.
- تصاویر نامعتبر یا خراب پیش از تقسیم داده حذف شده‌اند.

## ۳. معماری مدل اصلی
مدل اصلی دارای سه بلوک کانولوشنی با تعداد فیلترهای ۳۲، ۶۴ و ۱۲۸ است. هر بلوک شامل دو لایه Conv2D با کرنل ۳×۳، نرمال‌سازی Batch Normalization، تابع فعال‌ساز ReLU و MaxPooling است. پس از استخراج ویژگی‌ها، Global Average Pooling، یک لایه تمام‌متصل ۱۲۸ واحدی و خروجی دوکلاسه قرار دارد. Dropout برای کاهش بیش‌برازش به کار رفته است.

## ۴. روش آموزش
تابع هزینه Cross-Entropy، بهینه‌ساز AdamW، نرخ یادگیری اولیه **{learning_rate:g}**، batch size برابر **{batch_size}**، سقف آموزش **{epochs} epoch** و seed برابر **{seed}**، کاهش نرخ یادگیری بر اساس خطای اعتبارسنجی و Early Stopping استفاده شده‌اند. بهترین وزن‌ها بر اساس کمترین Validation Loss ذخیره شده‌اند.

## ۵. معیارهای ارزیابی روی مجموعه آزمون
در این گزارش، کلاس مثبت «سگ» در نظر گرفته شده است.

| معیار | مقدار |
|---|---:|
| Accuracy | {_pct(metrics['accuracy'])} |
| Precision (Dog) | {_pct(metrics['precision'])} |
| Recall (Dog) | {_pct(metrics['recall'])} |
| F1-score (Dog) | {_pct(metrics['f1_score'])} |
| MSE احتمالات (Brier score) | {metrics['mse']:.5f} |
| ROC-AUC | {_pct(metrics.get('roc_auc'))} |

### ماتریس ابهام
ترتیب سطرها و ستون‌ها به‌ترتیب **گربه، سگ** و سطرها نشان‌دهنده برچسب واقعی هستند.

| برچسب واقعی / پیش‌بینی | گربه | سگ |
|---|---:|---:|
| گربه | {matrix[0][0]} | {matrix[0][1]} |
| سگ | {matrix[1][0]} | {matrix[1][1]} |

نمودارهای MSE، روند یادگیری و ماتریس ابهام در پوشه `main/` ذخیره شده‌اند.

## ۶. بررسی اثر اجزای CNN
در آزمایش Ablation، تعداد بلوک‌های کانولوشنی (۲، ۳ و ۴ بلوک) و حضور Batch Normalization و Dropout تغییر داده شده‌اند. سایر شرایط از جمله تقسیم داده، اندازه تصویر، تعداد epochهای مجاز، batch size و نرخ یادگیری ثابت نگه داشته شده‌اند. این کار کمک می‌کند اثر عمق شبکه و روش‌های منظم‌سازی تا حد امکان جداگانه بررسی شود.

{ablation_table}

## ۷. تحلیل و بحث
- اختلاف Accuracy آموزش و اعتبارسنجی/آزمون نشانه‌ای از بیش‌برازش احتمالی است؛ نمودارهای `learning_curves.png` باید همراه این اختلاف بررسی شوند.
- اگر نسخه عمیق‌تر عملکرد اعتبارسنجی و آزمون را بهتر کند، افزایش ظرفیت شبکه برای این داده مفید بوده است؛ در غیر این صورت پیچیدگی بیشتر الزاماً به تعمیم بهتر منجر نشده است.
- افت عملکرد پس از حذف Batch Normalization یا Dropout می‌تواند نشان دهد این اجزا به پایداری آموزش یا کنترل بیش‌برازش کمک کرده‌اند. نتیجه باید بر اساس جدول واقعی آزمایش‌ها بیان شود، نه فرض قبلی.
- MSE در این پروژه میانگین مربع اختلاف احتمال پیش‌بینی کلاس سگ و برچسب صفر/یک است؛ بنابراین معیاری برای کیفیت احتمالات است و با Cross-Entropy که تابع هدف آموزش است تفاوت دارد.

## ۸. نتیجه‌گیری
مدل اصلی به Accuracy برابر **{_pct(metrics['accuracy'])}** و F1-score برابر **{_pct(metrics['f1_score'])}** روی مجموعه آزمون دست یافته است. نتیجه نهایی درباره بهترین معماری باید با رجوع به جدول Ablation و نمودارهای خروجی تکمیل شود.

## ۹. بازتولیدپذیری
تقسیم داده با seed ثابت ۴۲ انجام شده است. مسیرها و برچسب نمونه‌های هر بخش در فایل `data_splits.csv` ذخیره می‌شوند. وزن مدل، تاریخچه آموزش، معیارهای آزمون و نمودارها در پوشه `outputs/` نگهداری می‌شوند.

## منبع داده
Bhavik Jikadara, *Dog and Cat Classification Dataset*, Kaggle: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
'''
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
