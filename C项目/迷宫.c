#include<stdio.h>
#include<malloc.h>
#include<time.h>
#include<stdlib.h>
#include<windows.h>
#include<conio.h>

/*
必选问题：
程序中使用了哪种数据结构来解决问题？在代码中的变量名是什么？
其存储了什么信息？

答：1. 栈	node 存储了指针和drect结构体 
	2. 结构体	drect	存储了当前的位置和要走的方向	x,y,dir 

此种数据结构的特点（优点与缺点），你选择它的原因？

答：栈：先进后出， 具有记忆作用
	原因：要利用循环进行深度优先搜索 



*/


// 6个.h，这6个.h都是官方lib 
// 一个.c 

// 预定义 
typedef int datatype;

//位置
typedef struct drect 
{
	int x, y;
	int dir;
}drect;

//栈
typedef struct node
 {
	struct drect data;
	struct node* next;
}node;

//初始化一个栈
node* first()
{
	node* p = (node*)malloc(sizeof(node));
	p->next= NULL;
	return p;
}

//	压栈 
void push(node* s, int x, int y, int dir) 
{
	node* p =(node*)malloc(sizeof(node));
	p->data.x = x;
	p->data.y = y;
	p->data.dir = dir;
	p->next = s->next;
	s->next = p;
	//return s;
}


//出栈
node* output(node* s, int x, int y, int dir)
 {
	if (s== NULL)
		return 0;
	node* p ;
	p = s->next;
	s->next = p->next;
	return p;
}



//创建地图
void creatmap(int a[][100], int x, int y) 
{
	// 采用循环加随机数创造迷宫 
	int k;
	srand(time(0));
	for (int i = 0; i < x; i++) 
	{
		for (int j = 0; j < y; j++) 
		{
			if (i == 0 || i == x - 1)
			{
				a[i][j] = 1;
			}
			if (j == 0 || j == y - 1) 
			{
				a[i][j] = 1;
			}
			else if ((i != 0 && i != x - 1) && (j != 0 && j != y - 1)) 
			{
				a[i][j] = 0;
			}

		}
	}
	for (int i = 1; i < x - 1; i++) 
	{
		for (int j = 1; j < y - 1; j++) 
		{
			if (j < y / 3) 
			{//规范随机数个数产生数量
				k = rand() % (y - 1);
				a[i][k] = 1;
			}
		}
	}
}

//寻找出路
//right 1 under 2 left 3 on 4
int find(node*s , int a[][100], int w, int  h, int  x1, int y1, int x2, int y2, int x, int y)
{
	/*
	寻路算法采用的循环的方法，利用栈去实现的。2. 深度优先搜索 
	
	问题： 寻路是怎么实现的？
	答：    利用了深度优先搜索算法去实现的。
	
	问题：怎么去实现的深度优先搜索？
	答：利用循环加栈实现的深度优先搜索 
		
	*/ 
	node* p;
	int dir;// = 1;
	push(s, 1, 0, 1);
	while ((x!=x2||y!=y2)&& (x != x1 || y != y1))	// 循环的开始 
	{
		if ( a[x][y + 1] == 0) 
		{
			
			a[x][y] = -1;
			// 方向 
			dir = 1;
			// 入栈，对操作进行入栈，入栈出栈是算法的核心操作 
			push(s, x, y, dir); 
			// y++的作用是将现在的位置挪到了 x, y+1 
			y++;
			
		}
		else if ( a[x + 1][y] == 0) 
		{
			
			a[x][y] = -1;
			dir = 2;
			push(s, x, y, dir);
			// x++的作用就是将现在的位置挪到了 x+1, y 
			x++;
			
		}
		else if ( a[x][y - 1] == 0) 
		{
			a[x][y] = -1;
			dir = 3;
			push(s, x, y, dir);y--;
		}
		else if ( a[x - 1][y] == 0)
		{
			a[x][y] = -1;
			dir = 4;
			push(s, x, y, dir);x--;
		}
		else
		{
			// 这里指的是其他三个方向都已经走不了了。 一路出栈到可以走得通的时候，继续入栈寻路。 
			a[x][y] = -1;
			// 出栈 
			p = output(s, x, y, dir);
			if (p) 
			{
				x = p->data.x;
				y = p->data.y;
				dir = p->data.dir;
				// 出栈的时候，对指针进行空间释放，防止空间错误以及指针丢失 
				free(p);
			}
			
		}
		/*01
   	->0000011		
		  111	
		   1
		
		*/

	}
	if (x == x2 && y == y2)	// 到达了边界 
	{
		push(s, x, y, 0);
		return 1;	// 程序结束 
	}
	return 0;

}




void print_map(int a[][100],int w,int h) 
{
	// 循环加打印 出迷宫的步骤 
	printf("迷宫为：\n");
	for (int i = 0; i < w + 2; i++) 
	{
		for (int j = 0; j < h + 2; j++) 
		{
			a[1][0] = 0;
			a[1][1] = 0;
			a[w + 1][h] = 0;
			printf("%d ", a[i][j]);
		}
		printf("\n");
	}
}

void print_foot(node* s,int i)
/*
迷宫的出口打印 
*/ 
{
	if (i) 
	{
		printf("迷宫为：\n");
		while (s->next != NULL) 
		{
			s = s->next;
			int a = s->data.x;
			int b = s->data.y;
			int c = s->data.dir;
			// 整形打印 
			printf("(%d,%d)->%d\n", a, b, c);
		}
	}
	else
		printf("没有出口！\n");
}


///以下为动态迷宫游戏！！

int find_run_foot(char hyb[][200]) 
{
	int x = 1, y = 1;
	char ch;
	//hyb[1]=" ";
	for (int i = 0; i <= 20; i++)
	{
		puts(hyb[i]);
	}
	hyb[1][0] = ' ';
	while (1)
	{
		ch = _getch();
		if (ch == 's') 
		{
			if (hyb[x + 1][y] == ' ') 
			{
				hyb[x][y] = ' ';
				x++;
				hyb[x][y] = 'o';
			}
		}
		else if (ch == 'w') 
		{
			if (hyb[x - 1][y] == ' ') 
			{
				hyb[x][y] = ' ';
				x--;
				hyb[x][y] = 'o';
			}
		}
		else if (ch == 'a') 
		{
			if (hyb[x][y - 1] == ' ') 
			{
				hyb[x][y] = ' ';
				y--;
				hyb[x][y] = 'o';
			}
		}
		else if (ch == 'd')
		{
			if (hyb[x][y + 1] == ' ') 
			{
				hyb[x][y] = ' ';
				y++;
				hyb[x][y] = 'o';
			}
		}
		else {
			system("cls");
			printf("其他按键会退出游戏，你是否继续？1-是，其他-否\n");
			int dle;
			scanf("%d", &dle);
			system("cls");
			if (dle == 1) 
			{
				break;
			}
			else
			{
				system("cls");
				for (int i = 0; i <= 20; i++)
				{
					puts(hyb[i]);
				}
				continue;
			}
		}
		system("cls");
		for (int i = 0; i <= 20; i++)
		{
			puts(hyb[i]);
		}
		if (x == 19 && y == 27)
		{
			break;
		}
	}
	system("cls");
	Sleep(1000);
	printf("游戏结束！\n");
	system("cls");
	return 0;
}


int run_foot(int slect_slect) 
{
	
	/*捕获键盘信号进行游标*/
	char hyb1[200][200] = {
	"############################",
	"o  ### ## ### ## ###########",
	"##   ####          #########",
	"### #### ## ###### ###    ##",
	"### ###  ######### ### ## ##",
	"###         ###### ### ## ##",
	"### ## #### ######     ## ##",
	"#   ## ####   ########### ##",
	"# ####     ## ######      ##",
	"###    ### ##  ##### #######",
	"### ###### ###     #      ##",
	"#   ######    #### ###### ##",
	"# # ###### ####### #### # ##",
	"# #     ##      ## #### # ##",
	"# ##### ## #### ##        ##",
	"#    ## ## #### ######  # ##",
	"#### ## ## ##      #### # ##",
	"#    ## ##    ## ##   # #  #",
	"# ##### ## ##### ## # # ####",
	"#   #     ######    # #     ",
	"############################",
	};
	char hyb2[200][200] = {
		"############################################################################",
		"o    # ## ### #  ####     ##### ###### #### ####### ##### #### ## ### ######",
		"##    ##    ####        #         ###      #       ##### #### ## ### #######",
		"##### ## ### ### ##### ###  ## ## ### # ## ####### ##### #### ## ### ###   #",
		"##         # ### ##### ###  ## ## ### # ## ##            #### ## ### ###  ##",
		"## ## ## # # ### ##### #  #### ## ### #  # ## #### ### #   ## ## ### ### ###",
		"##### ## #     # ##### ###  ## ##   # ##  ## ## ##  ## ###        ## ### ###",
		"## ## ## ### # # ##        # ## ### #### ## ####    ### #### ## # # ###  ###",
		"## ## ## ### # # ## ## ### #                #### ###  ## #### ##   # #  ####",
		"## #####   ### # # ## ## #   # ###### ####       # ###  ## #### ## #   #####",
		"## ## ## ### #      ## #  ## ###### #### #######             # # #   #######",
		"## ## ## #  ## # # ###  ## ##  # ##   # #### ####### # ### #    ## # # #####",
		"## ##### ### ### #  #### ### # ## # # #### ##   ## ###  ## ## #    # # #####",
		"## ## ## ### ### ##### ### # ## # # #### ### ### ##### #  # # ## # # #######",
		"## ## ## ### #  ## ##### #   # ## # # #### ##   ## ####  # ##   ##      ####",
		"## ## ## ###  ## ##### ## ## ## # #        ###   ##   ####    ####  ####  ##",
		"## ## ## ##  #  ## ##### ## ## ## # # ## ##  ##  #####   #####         #####",
		"## #####   ### ### ##### ## ## ## # # #### #####   #        # ## ### #    ##",
		"## ## ###### ### ##    ## ## ## # # ####      ## ##  # # ## ##   # #      ##",
		"## ## ## ### ### ## ##  # ##    #   #### #### ## ## ## # ##     ## #####    ",
		"############################################################################",
	};
	char hyb[200][200];
	for (int i = 0; i < 200; i++) 
	{
		for (int j = 0; j < 200; j++) 
		{
			if (slect_slect == 1) 
			{
				hyb[i][j] = hyb1[i][j];
			}
			else if (slect_slect == 2) 
			{
				hyb[i][j] = hyb2[i][j];
			}
		}
	}
	find_run_foot(hyb);
	return 0;
}

//主函数 1. C语言的程序都是从main开始执行的 
int main()
{
	printf("-------欢迎来到迷宫小游戏------\n");
	printf("------------版本1.0------------\n");
	printf("说明：本游戏要时间加载，请耐心等待！\n");
	Sleep(500);	// 500ms 
	system("cls");	// clean list screen，清空屏幕 
	printf("等待游戏中………………\n");
	printf("游戏说明：用户请先初始化一个迷宫，然后根据自己需要是否要进行其他选择！\n");
	Sleep(1000);
	system("cls");
	// 声明指针定义 
	node* s;
	s = first();
	int	a[100][100];	// 存储迷宫的数组 
	int week;
	int w, h;
	int x1 = 1, y1 = 0,x2,y2, x = 1,  y = 1;
	int count = 0;
	int slect;
	while (1) 
	{
		printf("--------请选择版本型号----------\n");
		printf("--------1-静态迷宫1.0-----------\n");
		printf("--------2-动态迷宫1.0--------\n");
		printf("--------0-退出游戏-----------\n");
		printf("请输入版本型号：\n");
		// 输入你的选项 
		scanf("%d",&slect);
		system("cls");
		if (slect == 1) 
		{
			while (1) 
			{
				printf("-------1-初始化迷宫------------\n");
				printf("-------2-打印迷宫--------------\n");
				printf("-------3-出路迷宫--------------\n");
				printf("-------0-退出游戏--------------\n");
				printf("请输入你要选择的功能：\n");
				scanf("%d", &week);
				system("cls");
				if (week == 1) 
				{
					printf("请输入你要创建的迷宫大小：\n");
					// 接受两个整形  a b 默认的情况下我们要给迷宫算法设置一道墙， 
					scanf("%d%d", &w, &h);
					// 创造迷宫 
					creatmap(a, w + 2, h + 2);
					count++;
				}
				else if (week == 2) 
				{
					if (count > 0) 
					{
						print_map(a, w, h);
						count++;
					}
					else 
					{
						printf("请先初始化一个迷宫！\n");
						Sleep(1000);
						system("cls");
						continue;
					}
				}
				else if (week == 3) 
				{
					if (count > 0) 
					{
						x2 = w + 1;
						y2 = h;
						// find是寻路算法 
						int i = find(s, a, w, h, x1, y1, x2, y2, x, y);
						print_foot(s, i);
						count++;
					}
					else 
					{
						printf("请先初始化一个迷宫！\n");
						Sleep(1000);
						system("cls");
						continue;
					}
				}
				else if (week == 0) 
				{
					system("cls");
					printf("---------------游戏结束！--------------\n");
					Sleep(1000);
					break;
				}
				int m;
				printf("请选择是否执行程序!\n");
				printf("1-true,0-fasle\n");
				scanf("%d", &m);
				system("cls");
				if (m == 1) 
				{
					continue;
				}
				else
					system("cls");
				printf("---------------游戏结束！--------------\n");
				Sleep(1000);
				break;
			}
		}
		else if (slect == 2) 
		{
			int slect1;
			printf("游戏说明：该游戏控制的方向如下：w（上）s（下）a（左）d（右）。\n若无法移动，可按住alt键在进行操作！\n");
			Sleep(1000);
			system("cls");
			printf("------------请输入选项！----------\n");
			printf("------------1-简单模式------------\n");
			printf("------------2-困难模式------------\n");
			printf("			……				  \n");
			printf("------------0-退出游戏------------\n");
			scanf("%d",&slect1);
			system("cls");
			if (slect1 == 1) 
			{
				run_foot(1);
				continue;
			}
			else if (slect1 == 2) 
			{
				run_foot(2);
				continue;
			}
			else if (slect1 == 0) 
			{
				continue;
			}
			else 
			{
				printf("-----其他选项待开发！----");
				Sleep(1000);
				continue;
			}
		}
		else if(slect==0)
		{
			break;
		}
		else 
		{
			printf("其他大型迷宫待开发中！");
			continue;
		}
	}
	return 0;
}

