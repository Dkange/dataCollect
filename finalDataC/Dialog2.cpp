// Dialog2.cpp: 实现文件
//

#include "pch.h"
#include "finalDataC.h"
#include "Dialog2.h"
#include "afxdialogex.h"


// Dialog2 对话框

IMPLEMENT_DYNAMIC(Dialog2, CDialogEx)

Dialog2::Dialog2(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG2, pParent)
{

}

Dialog2::~Dialog2()
{
}

void Dialog2::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(Dialog2, CDialogEx)
END_MESSAGE_MAP()


// Dialog2 消息处理程序
