import pytest
import pandas as pd


@pytest.fixture
def sample_career_data():
    """
    Dữ liệu mẫu về nhóm nghề nghiệp.
    """

    data = {
        "MaNhom": ["N1", "N2", "N3", "N4"],

        "NhomNghe": [
            "Công nghệ – Kỹ thuật – Phân tích",
            "Kinh doanh – Tài chính – Quản lý",
            "Sáng tạo – Truyền thông – Thiết kế",
            "Giáo dục – Xã hội – Hỗ trợ con người"
        ],

        "NganhDaoTao": [
            "Công nghệ thông tin, khoa học dữ liệu, điện–điện tử, tự động hóa, cơ khí",
            "Quản trị kinh doanh, tài chính–ngân hàng, kế toán, kinh tế, logistics",
            "Thiết kế đồ họa, truyền thông đa phương tiện, báo chí, quảng cáo",
            "Sư phạm, tâm lý học, công tác xã hội, quản trị nhân lực, điều dưỡng"
        ],

        "NgheGoiY": [
            "Lập trình viên, kiểm thử phần mềm, chuyên viên phân tích dữ liệu, kỹ sư điện",
            "Chuyên viên kinh doanh, phân tích tài chính, kế toán viên, chuyên viên logistics",
            "Thiết kế đồ họa, thiết kế UI/UX, sáng tạo nội dung, biên tập viên",
            "Giáo viên, chuyên viên đào tạo, chuyên viên tuyển dụng, điều dưỡng"
        ]
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_student_data():
    """
    Dữ liệu mẫu của người dùng làm bài đánh giá.
    Bao gồm 8 yếu tố năng lực và 6 yếu tố sở thích RIASEC.
    """

    data = {
        "HoTen": [
            "Nguyễn Văn A",
            "Trần Thị B",
            "Lê Văn C",
            "Phạm Thị D"
        ],

        # 8 yếu tố năng lực
        "TuDuyLogic": [5, 4, 3, 3],
        "TinhToan_PhanTich": [5, 4, 3, 3],
        "CongNghe": [5, 3, 4, 2],
        "SangTao": [3, 3, 5, 4],
        "GiaoTiep": [3, 5, 4, 5],
        "ToChuc": [4, 5, 3, 4],
        "CanThan": [5, 5, 4, 4],
        "DongCam": [3, 3, 4, 5],

        # 6 yếu tố RIASEC
        "RIASEC_R_Thuc_hanh": [4, 2, 2, 2],
        "RIASEC_I_Nghien_cuu": [5, 4, 3, 4],
        "RIASEC_A_Nghe_thuat": [2, 3, 5, 4],
        "RIASEC_S_Xa_hoi": [2, 3, 4, 5],
        "RIASEC_E_Thuyet_phuc": [3, 5, 4, 3],
        "RIASEC_C_Quy_cu": [4, 5, 3, 4]
    }

    return pd.DataFrame(data)


@pytest.fixture
def invalid_student_data():
    """
    Dữ liệu không hợp lệ để kiểm thử validate.
    Có điểm vượt ngoài thang điểm 1-5.
    """

    data = {
        "HoTen": ["Người dùng lỗi"],
        "TuDuyLogic": [6],  # Sai: chỉ được từ 1 đến 5
        "TinhToan_PhanTich": [0],  # Sai
        "CongNghe": [5],
        "SangTao": [3],
        "GiaoTiep": [4],
        "ToChuc": [4],
        "CanThan": [5],
        "DongCam": [3],
        "RIASEC_R_Thuc_hanh": [4],
        "RIASEC_I_Nghien_cuu": [5],
        "RIASEC_A_Nghe_thuat": [3],
        "RIASEC_S_Xa_hoi": [4],
        "RIASEC_E_Thuyet_phuc": [3],
        "RIASEC_C_Quy_cu": [4]
    }

    return pd.DataFrame(data)
