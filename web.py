import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
import tempfile

# =========================================
# إعدادات الصفحة
# =========================================
st.set_page_config(
    page_title="DAT Land Cover Classifier",
    layout="wide"
)

st.title("🛰 Land Cover Classifier")

st.write("يدعم ملفات DAT فقط")

# =========================================
# تحميل الموديل والـ Scaler
# =========================================
@st.cache_resource
def load_ai_assets():

    model_path = "best_land_cover_model.joblib"
    scaler_path = "data_scaler.pkl"

    if not os.path.exists(model_path):
        st.error("❌ ملف الموديل غير موجود")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error("❌ ملف الـ scaler غير موجود")
        st.stop()

    try:

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        return model, scaler

    except Exception as e:

        st.error(f"❌ خطأ أثناء تحميل الملفات: {e}")
        st.stop()

model, scaler = load_ai_assets()

# =========================================
# رفع ملف DAT
# =========================================
uploaded_file = st.file_uploader(
    "Upload STACK.dat",
    type=["dat"]
)

# =========================================
# بدء التنفيذ
# =========================================
if uploaded_file is not None:

    st.success("✔️ تم تحميل ملف الـ DAT")

    if st.button("🚀 Run Full Pipeline"):

        try:

            st.info("⏳ جاري قراءة ملف الـ DAT...")

            # =====================================
            # حفظ الملف مؤقتًا
            # =====================================
            with tempfile.TemporaryDirectory() as tmpdir:

                dat_path = os.path.join(
                    tmpdir,
                    uploaded_file.name
                )

                with open(dat_path, "wb") as f:
                    f.write(uploaded_file.read())

                # =====================================
                # قراءة الملف الخام
                # =====================================
                raw = np.fromfile(
                    dat_path,
                    dtype=np.float32
                )

            # =====================================
            # الأبعاد الثابتة
            # =====================================
            BANDS = 10
            ROWS = 2000
            COLS = 2000

            expected_size = (
                BANDS
                * ROWS
                * COLS
            )

            # =====================================
            # التحقق من الحجم
            # =====================================
            if raw.size != expected_size:

                st.error(
                    f"❌ حجم الملف غير صحيح\n\n"
                    f"Expected: {expected_size}\n"
                    f"Found: {raw.size}"
                )

                st.stop()

            # =====================================
            # إعادة تشكيل البيانات
            # =====================================
            full_features = raw.reshape(
                BANDS,
                ROWS,
                COLS
            )

            # =====================================
            # تجهيز البيانات للموديل
            # =====================================
            flat_data = full_features.reshape(
                10,
                -1
            ).T

            flat_data = np.nan_to_num(
                flat_data
            )

            st.info("⏳ جاري الـ Scaling والتصنيف...")

            # =====================================
            # Scaling
            # =====================================
            flat_scaled = scaler.transform(
                flat_data
            )

            # =====================================
            # Prediction
            # =====================================
            preds = model.predict(
                flat_scaled
            )

            classification_map = preds.reshape(
                ROWS,
                COLS
            ).astype(float)

            # =====================================
            # إزالة المثلث الأيسر
            # =====================================
            for r in range(1400):

                limit = int(
                    260 - (r * 0.11)
                )

                if limit > 0:

                    classification_map[
                        r,
                        :limit
                    ] = np.nan

            # =====================================
            # عرض النتائج
            # =====================================
            col1, col2 = st.columns([3, 2])

            # =====================================
            # عرض الخريطة
            # =====================================
            with col1:

                st.subheader(
                    "🗺️ Classification Map"
                )

                cmap = ListedColormap([
                    "blue",
                    "green",
                    "yellow",
                    "red"
                ])

                cmap.set_bad(
                    color="white"
                )

                fig, ax = plt.subplots(
                    figsize=(10, 10)
                )

                ax.imshow(
                    classification_map,
                    cmap=cmap,
                    vmin=0,
                    vmax=3
                )

                ax.axis("off")

                st.pyplot(fig)

            # =====================================
            # الإحصائيات
            # =====================================
            with col2:

                st.subheader(
                    "📊 Area Statistics"
                )

                pixel_area_km2 = (
                    30 * 30
                ) / 1_000_000.0

                labels_map = {
                    0: "Water 🟦",
                    1: "Vegetation 🟩",
                    2: "Bare Soil / Desert 🟨",
                    3: "Urban / Built-up 🟥"
                }

                stats_list = []

                for val, name in labels_map.items():

                    count = np.sum(
                        classification_map == val
                    )

                    area = (
                        count
                        * pixel_area_km2
                    )

                    stats_list.append({
                        "Class Name": name,
                        "Pixel Count": int(count),
                        "Area (km²)": round(area, 2)
                    })

                df_stats = pd.DataFrame(
                    stats_list
                )

                total_area = df_stats[
                    "Area (km²)"
                ].sum()

                df_stats[
                    "Percentage (%)"
                ] = round(
                    (
                        df_stats["Area (km²)"]
                        / total_area
                    ) * 100,
                    2
                )

                st.dataframe(
                    df_stats,
                    use_container_width=True
                )

                # =================================
                # الرسم البياني
                # =================================
                st.write(
                    "### Class Distribution"
                )

                fig_bar, ax_bar = plt.subplots(
                    figsize=(6, 4)
                )

                ax_bar.bar(
                    df_stats["Class Name"],
                    df_stats["Percentage (%)"],
                    color=[
                        "blue",
                        "green",
                        "gold",
                        "red"
                    ]
                )

                ax_bar.set_ylabel(
                    "Percentage (%)"
                )

                plt.xticks(
                    rotation=15
                )

                st.pyplot(fig_bar)

            st.success(
                "✅ اكتمل التصنيف بنجاح"
            )

        except Exception as e:

            st.error(
                f"❌ حدث خطأ أثناء المعالجة: {e}"
            )
