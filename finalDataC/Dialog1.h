#include <opencv2/opencv.hpp>
#pragma once
#define POINT_COUNT 100 //显示的波形的长度

// Dialog1 对话框

class Dialog1 : public CDialogEx
{
	DECLARE_DYNAMIC(Dialog1)

public:
	Dialog1(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~Dialog1();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG1 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
private:
	CEdit m_edit1;
	CEdit m_edit2;
	CEdit m_edit3;
	CEdit m_len1;
	CEdit m_len2;
	CEdit m_len3;
	CStatic m_draw1;
	CStatic m_draw2;
	CStatic m_draw3;
public:
	void Draw1(CDC* pDC, CRect& rectPicture, float* m_value);
	float m_value1[POINT_COUNT];
	float m_value2[POINT_COUNT];
	float m_value3[POINT_COUNT];
	float* len, * angle;//全局变量 肢体长度和关节角度
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
};
