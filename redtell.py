import tkinter as tk
from tkinter import filedialog, messagebox
from segmentation.predict import segment_images
from segmentation.train import train_new_model
from feature_extraction.extract_features import extract_features
from annotation.generate_annotations import create_annotation_cells


# ایجاد رابط گرافیکی با استفاده از Tkinter
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("رابط گرافیکی پردازش داده")

        # ساختن ورودی‌ها برای انتخاب نوع عملکرد
        self.funct_label = tk.Label(root, text="انتخاب عملکرد:")
        self.funct_label.grid(row=0, column=0, pady=10)

        self.funct_var = tk.StringVar()
        self.funct_var.set("segment")  # مقدار پیشفرض
        self.funct_options = ["segment", "train_segmentation", "extract_features", "annotate"]
        self.funct_menu = tk.OptionMenu(root, self.funct_var, *self.funct_options)
        self.funct_menu.grid(row=0, column=1, pady=10)

        # انتخاب مسیر داده
        self.data_label = tk.Label(root, text="انتخاب مسیر داده:")
        self.data_label.grid(row=1, column=0, pady=10)

        self.data_button = tk.Button(root, text="انتخاب فایل یا پوشه", command=self.select_data)
        self.data_button.grid(row=1, column=1, pady=10)

        # ورودی مدل (اختیاری)
        self.model_label = tk.Label(root, text="مدل (اختیاری):")
        self.model_label.grid(row=2, column=0, pady=10)

        self.model_entry = tk.Entry(root)
        self.model_entry.grid(row=2, column=1, pady=10)

        # ورودی کانال‌ها برای استخراج ویژگی‌ها
        self.channel_label = tk.Label(root, text="کانال‌ها (اختیاری):")
        self.channel_label.grid(row=3, column=0, pady=10)

        self.channel_entry = tk.Entry(root)
        self.channel_entry.grid(row=3, column=1, pady=10)

        # ورودی تعداد سلول‌ها برای حاشیه‌نویسی
        self.num_cells_label = tk.Label(root, text="تعداد سلول‌ها (اختیاری):")
        self.num_cells_label.grid(row=4, column=0, pady=10)

        self.num_cells_entry = tk.Entry(root)
        self.num_cells_entry.grid(row=4, column=1, pady=10)

        # دکمه اجرا
        self.run_button = tk.Button(root, text="اجرای انتخاب‌ها", command=self.run_function)
        self.run_button.grid(row=5, column=0, columnspan=2, pady=20)

    def select_data(self):
        # انتخاب مسیر داده توسط کاربر
        self.data_path = filedialog.askdirectory(title="انتخاب پوشه داده")
        if not self.data_path:
            self.data_path = filedialog.askopenfilename(title="انتخاب فایل داده")
        print(f"مسیر داده انتخاب‌شده: {self.data_path}")

    def run_function(self):
        # گرفتن مقادیر واردشده توسط کاربر
        funct = self.funct_var.get()
        data = self.data_path
        model = self.model_entry.get() if self.model_entry.get() else None
        channels = self.channel_entry.get().split() if self.channel_entry.get() else None
        num_cells = int(self.num_cells_entry.get()) if self.num_cells_entry.get() else None

        # اجرا کردن تابع بر اساس انتخاب کاربر
        if funct == "segment":
            if model is None:
                model = "mask_rcnn_commitment"
            segment_images(data, model)

        elif funct == "train_segmentation":
            train_new_model(data, model, num_epochs=15)

        elif funct == "extract_features":
            if channels is None:
                channels = ["mask", "bf"]
            extract_features(data, channels)

        elif funct == "annotate":
            if num_cells is None:
                num_cells = 200
            create_annotation_cells(data, num_cells)

        else:
            messagebox.showerror("خطا", "عملکرد مشخص‌شده معتبر نیست.")

        # نمایش پیامی برای نشان دادن اتمام عملیات
        messagebox.showinfo("اطلاع", "عملیات با موفقیت انجام شد!")


# ایجاد و نمایش پنجره اصلی
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
