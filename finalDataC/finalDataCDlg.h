
// finalDataCDlg.h: 头文件
//

#pragma once
#include "TabSheet.h"
#include "Dialog1.h"
#include "Dialog2.h"

// CfinalDataCDlg 对话框
class CfinalDataCDlg : public CDialogEx
{
// 构造
public:
	CfinalDataCDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_FINALDATAC_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
private:
	CTabSheet m_tabCtrl;
	Dialog1 dlg1;
	Dialog2 dlg2;
};
