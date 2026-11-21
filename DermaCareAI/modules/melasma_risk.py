"""
melasma_risk.py -- Mô hình đánh giá nguy cơ mắc bệnh nám má (Melasma Risk Assessment)
dựa trên nghiên cứu lâm sàng của BV Phong - Da liễu Trung ương Quy Hòa:
"Applying Machine Learning to Predict Melasma" (Hồ Văn Lâm, Vũ Tuấn Anh, Phạm Thị Hoàng Bích Dịu, Trần Xuân Việt, 2021).

Nghiên cứu trên 795 mẫu khảo sát (238 ca mắc nám, 557 ca không mắc) với 21 đặc trưng lâm sàng và thói quen.
Các biến số có tầm ảnh hưởng lớn nhất theo XGBoost Feature Importance & Multivariable Logistic Regression:
  1. Uống thuốc tránh thai (Birth control pills): 9.5% tầm ảnh hưởng
  2. Tiền sử gia đình mắc nám (Family history): 9.3%
  3. Hoàn cảnh kinh tế gia đình (Family Economy): 8.8% (Nhóm nghèo/cận nghèo có nguy cơ cao gấp 3.91 lần)
  4. Tháng mang thai / thai kỳ (Month of pregnancy): 8.7%
  5. Vùng địa lý sinh sống (Location): 8.6%
  6. Từng bị nám khi mang thai (Melasma during pregnancy): 6.2% (Nguy cơ cao gấp 2.93 lần)
  7. Bệnh lý mạn tính kèm theo (Pathology): 5.1%
  8. Năm sinh / Độ tuổi (Age): 4.0% (Nhóm tuổi 35 - 45 có nguy cơ cao nhất)
  9. Phơi nắng buổi chiều (Sun exposure afternoon): 4.0%
 10. Số lần mang thai (Number of pregnancies): 3.8%
 11. Phơi nắng buổi trưa (Sun exposure noon): 3.7%
 12. Phơi nắng buổi sáng (Sun exposure morning): 3.7%
 13. Tổng số giờ phơi nắng hàng ngày: 3.7%
 14. Độ tuổi bắt đầu dùng mỹ phẩm: 3.5%
 15. Dùng mỹ phẩm làm trắng da (Whitening cosmetics): Tăng nguy cơ 1.5 lần
 16. Nghề nghiệp, học vấn, tiếp xúc hóa chất, số lần sinh con, tình trạng hôn nhân...
"""

def evaluate_risk(data):
    """
    Tính toán xác suất nguy cơ mắc nám dựa trên mô hình điểm trọng số từ nghiên cứu BV Quy Hòa.
    data: dict chứa các trường khảo sát từ người dùng.
    Trả về dict: {
        'risk_score': float (0 - 100%),
        'risk_level': 'Thấp' | 'Trung bình' | 'Cao',
        'odds_ratio_estimate': float,
        'primary_factors': list of str,
        'recommendations': list of str,
        'suggest_image_analysis': bool
    }
    """
    score = 0.0
    primary_factors = []
    recommendations = []

    # 1. Tuổi (Age) - Nhóm 35-45 có tỷ lệ mắc cao nhất trong nghiên cứu (Hình 1)
    try:
        age = int(data.get("age", 30))
    except (ValueError, TypeError):
        age = 30
    if 35 <= age <= 45:
        score += 8.0
        primary_factors.append("Độ tuổi 35–45 (giai đoạn biến động nội tiết tố mạnh nhất)")
    elif 25 <= age < 35 or 45 < age <= 55:
        score += 5.0
    elif age > 55:
        score += 3.0

    # 2. Uống thuốc tránh thai (Birth control pills) - 9.5% tầm ảnh hưởng
    pills = str(data.get("birth_control_pills", "no")).lower() in ("yes", "true", "1", "có")
    if pills:
        score += 15.0
        primary_factors.append("Sử dụng thuốc tránh thai đường uống (chứa estrogen/progestin kích thích tế bào hắc tố)")
        recommendations.append("Nên tham khảo ý kiến bác sĩ phụ khoa/da liễu để cân nhắc đổi sang biện pháp tránh thai không dùng hormone nội tiết.")

    # 3. Tiền sử gia đình mắc nám (Family history) - 9.3% tầm ảnh hưởng
    fam = str(data.get("family_history", "no")).lower() in ("yes", "true", "1", "có")
    if fam:
        score += 14.0
        primary_factors.append("Có tiền sử người thân trong gia đình mắc nám (yếu tố di truyền học)")

    # 4. Tiền sử nám khi mang thai (Melasma during pregnancy) - OR = 2.93 (Tăng nguy cơ 2.93 lần)
    preg_mel = str(data.get("melasma_during_pregnancy", "no")).lower() in ("yes", "true", "1", "có")
    if preg_mel:
        score += 13.0
        primary_factors.append("Từng xuất hiện đốm nâu/nám trong giai đoạn mang thai (nám nhạy cảm với hormone thai kỳ)")

    # 5. Tình trạng mang thai hiện tại hoặc số lần sinh con
    is_pregnant = str(data.get("currently_pregnant", "no")).lower() in ("yes", "true", "1", "có")
    if is_pregnant:
        score += 8.0
        primary_factors.append("Đang trong giai đoạn mang thai (nồng độ MSH, estrogen và progesterone tăng cao)")

    # 6. Tiếp xúc ánh nắng mặt trời (Sun exposure) - Cường độ trưa & chiều đặc biệt nguy hại
    sun_hours = float(data.get("sun_hours", 2))
    sun_noon = str(data.get("sun_noon", "no")).lower() in ("yes", "true", "1", "có")
    sun_afternoon = str(data.get("sun_afternoon", "no")).lower() in ("yes", "true", "1", "có")

    sun_score = 0.0
    if sun_hours >= 4:
        sun_score += 8.0
    elif sun_hours >= 2:
        sun_score += 4.0
    
    if sun_noon:  # 11h - 14h: Đỉnh điểm tia UVA & UVB
        sun_score += 6.0
        primary_factors.append("Thường xuyên tiếp xúc ánh nắng vào khung giờ trưa (11h–14h, tia UV đạt cực đại)")
    if sun_afternoon: # 14h - 17h
        sun_score += 4.0
        primary_factors.append("Tiếp xúc ánh nắng buổi chiều (tia UVA tích lũy gây tổn thương sâu đáy biểu bì)")
    
    score += min(16.0, sun_score)
    if sun_score > 6.0:
        recommendations.append("Bắt buộc sử dụng kem chống nắng phổ rộng (Broad Spectrum, SPF 50+, PA++++) hàng ngày, thoa lại sau mỗi 2–3 giờ khi hoạt động ngoài trời.")

    # 7. Dùng mỹ phẩm làm trắng da (Whitening cosmetics) - Nguy cơ tăng 1.5 lần
    cosmetics = str(data.get("whitening_cosmetics", "no")).lower() in ("yes", "true", "1", "có")
    if cosmetics:
        score += 9.0
        primary_factors.append("Sử dụng mỹ phẩm làm trắng/lột tẩy da (nguy cơ bào mòn hàng rào bảo vệ da, tăng sắc tố sau viêm)")
        recommendations.append("Ngừng ngay các loại mỹ phẩm làm trắng cấp tốc hoặc kem trộn không rõ nguồn gốc (dễ chứa Corticoid hoặc Hydroquinone nồng độ cao gây mất sắc tố không hồi phục).")

    # 8. Yếu tố hoàn cảnh kinh tế & Môi trường làm việc ngoài trời
    outdoor_work = str(data.get("outdoor_work", "no")).lower() in ("yes", "true", "1", "có")
    if outdoor_work:
        score += 7.0
        primary_factors.append("Tính chất công việc ngoài trời hoặc thường xuyên tiếp xúc môi trường nhiệt độ cao/bụi bẩn")
        recommendations.append("Trang bị đồ bảo hộ chống nắng vật lý: mũ rộng vành, khẩu trang vải dày dệt sợi chống tia UV, kính râm khi làm việc ngoài trời.")

    # 9. Tiếp xúc hóa chất hoặc bệnh lý mạn tính (gan, tuyến giáp...)
    chronic_pathology = str(data.get("chronic_pathology", "no")).lower() in ("yes", "true", "1", "có")
    if chronic_pathology:
        score += 6.0
        primary_factors.append("Có bệnh lý mạn tính kèm theo (rối loạn chức năng gan, tuyến giáp hoặc dùng thuốc dài ngày)")

    # Chuẩn hóa về thang 0 - 100%
    risk_percent = min(96.0, max(5.0, round(score * 1.05, 1)))

    if risk_percent >= 60.0:
        level = "Cao"
        level_class = "danger"
        advice = "Bạn có nguy cơ mắc hoặc tiến triển nám da ở mức độ CAO. Nên sớm thăm khám với bác sĩ chuyên khoa da liễu và tiến hành phân tích hình thái khuôn mặt để đánh giá diện tích tổn thương."
    elif risk_percent >= 30.0:
        level = "Trung bình"
        level_class = "warning"
        advice = "Bạn có nguy cơ mắc nám ở mức TRUNG BÌNH. Các yếu tố sinh hoạt và môi trường đang bắt đầu tác động lên làn da. Hãy điều chỉnh thói quen chống nắng và chăm sóc da ngay."
    else:
        level = "Thấp"
        level_class = "success"
        advice = "Nguy cơ mắc nám hiện tại của bạn ở mức THẤP. Hãy duy trì thói quen chống nắng và chăm sóc da khoa học."

    if not recommendations:
        recommendations.append("Duy trì bôi kem chống nắng đều đặn, dưỡng ẩm và uống đủ 2 lít nước mỗi ngày.")

    return {
        "success": True,
        "risk_percent": risk_percent,
        "risk_level": level,
        "risk_level_class": level_class,
        "general_advice": advice,
        "primary_factors": primary_factors,
        "recommendations": recommendations,
        "suggest_image_analysis": risk_percent >= 30.0,
        "study_source": "Nghiên cứu mô hình Machine Learning dự đoán nám má - BV Phong Da Liễu Quy Hòa (Hồ Văn Lâm và cs., IJCSIS 2021)"
    }
