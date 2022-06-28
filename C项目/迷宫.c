#include<stdio.h>
#include<malloc.h>
#include<time.h>
#include<stdlib.h>
#include<windows.h>
#include<conio.h>

/*
��ѡ���⣺
������ʹ�����������ݽṹ��������⣿�ڴ����еı�������ʲô��
��洢��ʲô��Ϣ��

��1. ջ	node �洢��ָ���drect�ṹ�� 
	2. �ṹ��	drect	�洢�˵�ǰ��λ�ú�Ҫ�ߵķ���	x,y,dir 

�������ݽṹ���ص㣨�ŵ���ȱ�㣩����ѡ������ԭ��

��ջ���Ƚ������ ���м�������
	ԭ��Ҫ����ѭ����������������� 



*/


// 6��.h����6��.h���ǹٷ�lib 
// һ��.c 

// Ԥ���� 
typedef int datatype;

//λ��
typedef struct drect 
{
	int x, y;
	int dir;
}drect;

//ջ
typedef struct node
 {
	struct drect data;
	struct node* next;
}node;

//��ʼ��һ��ջ
node* first()
{
	node* p = (node*)malloc(sizeof(node));
	p->next= NULL;
	return p;
}

//	ѹջ 
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


//��ջ
node* output(node* s, int x, int y, int dir)
 {
	if (s== NULL)
		return 0;
	node* p ;
	p = s->next;
	s->next = p->next;
	return p;
}



//������ͼ
void creatmap(int a[][100], int x, int y) 
{
	// ����ѭ��������������Թ� 
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
			{//�淶�����������������
				k = rand() % (y - 1);
				a[i][k] = 1;
			}
		}
	}
}

//Ѱ�ҳ�·
//right 1 under 2 left 3 on 4
int find(node*s , int a[][100], int w, int  h, int  x1, int y1, int x2, int y2, int x, int y)
{
	/*
	Ѱ·�㷨���õ�ѭ���ķ���������ջȥʵ�ֵġ�2. ����������� 
	
	���⣺ Ѱ·����ôʵ�ֵģ�
	��    ������������������㷨ȥʵ�ֵġ�
	
	���⣺��ôȥʵ�ֵ��������������
	������ѭ����ջʵ�ֵ������������ 
		
	*/ 
	node* p;
	int dir;// = 1;
	push(s, 1, 0, 1);
	while ((x!=x2||y!=y2)&& (x != x1 || y != y1))	// ѭ���Ŀ�ʼ 
	{
		if ( a[x][y + 1] == 0) 
		{
			
			a[x][y] = -1;
			// ���� 
			dir = 1;
			// ��ջ���Բ���������ջ����ջ��ջ���㷨�ĺ��Ĳ��� 
			push(s, x, y, dir); 
			// y++�������ǽ����ڵ�λ��Ų���� x, y+1 
			y++;
			
		}
		else if ( a[x + 1][y] == 0) 
		{
			
			a[x][y] = -1;
			dir = 2;
			push(s, x, y, dir);
			// x++�����þ��ǽ����ڵ�λ��Ų���� x+1, y 
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
			// ����ָ�����������������Ѿ��߲����ˡ� һ·��ջ�������ߵ�ͨ��ʱ�򣬼�����ջѰ·�� 
			a[x][y] = -1;
			// ��ջ 
			p = output(s, x, y, dir);
			if (p) 
			{
				x = p->data.x;
				y = p->data.y;
				dir = p->data.dir;
				// ��ջ��ʱ�򣬶�ָ����пռ��ͷţ���ֹ�ռ�����Լ�ָ�붪ʧ 
				free(p);
			}
			
		}
		/*01
   	->0000011		
		  111	
		   1
		
		*/

	}
	if (x == x2 && y == y2)	// �����˱߽� 
	{
		push(s, x, y, 0);
		return 1;	// ������� 
	}
	return 0;

}




void print_map(int a[][100],int w,int h) 
{
	// ѭ���Ӵ�ӡ ���Թ��Ĳ��� 
	printf("�Թ�Ϊ��\n");
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
�Թ��ĳ��ڴ�ӡ 
*/ 
{
	if (i) 
	{
		printf("�Թ�Ϊ��\n");
		while (s->next != NULL) 
		{
			s = s->next;
			int a = s->data.x;
			int b = s->data.y;
			int c = s->data.dir;
			// ���δ�ӡ 
			printf("(%d,%d)->%d\n", a, b, c);
		}
	}
	else
		printf("û�г��ڣ�\n");
}


///����Ϊ��̬�Թ���Ϸ����

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
			printf("�����������˳���Ϸ�����Ƿ������1-�ǣ�����-��\n");
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
	printf("��Ϸ������\n");
	system("cls");
	return 0;
}


int run_foot(int slect_slect) 
{
	
	/*��������źŽ����α�*/
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

//������ 1. C���Եĳ����Ǵ�main��ʼִ�е� 
int main()
{
	printf("-------��ӭ�����Թ�С��Ϸ------\n");
	printf("------------�汾1.0------------\n");
	printf("˵��������ϷҪʱ����أ������ĵȴ���\n");
	Sleep(500);	// 500ms 
	system("cls");	// clean list screen�������Ļ 
	printf("�ȴ���Ϸ�С�����������\n");
	printf("��Ϸ˵�����û����ȳ�ʼ��һ���Թ���Ȼ������Լ���Ҫ�Ƿ�Ҫ��������ѡ��\n");
	Sleep(1000);
	system("cls");
	// ����ָ�붨�� 
	node* s;
	s = first();
	int	a[100][100];	// �洢�Թ������� 
	int week;
	int w, h;
	int x1 = 1, y1 = 0,x2,y2, x = 1,  y = 1;
	int count = 0;
	int slect;
	while (1) 
	{
		printf("--------��ѡ��汾�ͺ�----------\n");
		printf("--------1-��̬�Թ�1.0-----------\n");
		printf("--------2-��̬�Թ�1.0--------\n");
		printf("--------0-�˳���Ϸ-----------\n");
		printf("������汾�ͺţ�\n");
		// �������ѡ�� 
		scanf("%d",&slect);
		system("cls");
		if (slect == 1) 
		{
			while (1) 
			{
				printf("-------1-��ʼ���Թ�------------\n");
				printf("-------2-��ӡ�Թ�--------------\n");
				printf("-------3-��·�Թ�--------------\n");
				printf("-------0-�˳���Ϸ--------------\n");
				printf("��������Ҫѡ��Ĺ��ܣ�\n");
				scanf("%d", &week);
				system("cls");
				if (week == 1) 
				{
					printf("��������Ҫ�������Թ���С��\n");
					// ������������  a b Ĭ�ϵ����������Ҫ���Թ��㷨����һ��ǽ�� 
					scanf("%d%d", &w, &h);
					// �����Թ� 
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
						printf("���ȳ�ʼ��һ���Թ���\n");
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
						// find��Ѱ·�㷨 
						int i = find(s, a, w, h, x1, y1, x2, y2, x, y);
						print_foot(s, i);
						count++;
					}
					else 
					{
						printf("���ȳ�ʼ��һ���Թ���\n");
						Sleep(1000);
						system("cls");
						continue;
					}
				}
				else if (week == 0) 
				{
					system("cls");
					printf("---------------��Ϸ������--------------\n");
					Sleep(1000);
					break;
				}
				int m;
				printf("��ѡ���Ƿ�ִ�г���!\n");
				printf("1-true,0-fasle\n");
				scanf("%d", &m);
				system("cls");
				if (m == 1) 
				{
					continue;
				}
				else
					system("cls");
				printf("---------------��Ϸ������--------------\n");
				Sleep(1000);
				break;
			}
		}
		else if (slect == 2) 
		{
			int slect1;
			printf("��Ϸ˵��������Ϸ���Ƶķ������£�w���ϣ�s���£�a����d���ң���\n���޷��ƶ����ɰ�סalt���ڽ��в�����\n");
			Sleep(1000);
			system("cls");
			printf("------------������ѡ�----------\n");
			printf("------------1-��ģʽ------------\n");
			printf("------------2-����ģʽ------------\n");
			printf("			����				  \n");
			printf("------------0-�˳���Ϸ------------\n");
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
				printf("-----����ѡ���������----");
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
			printf("���������Թ��������У�");
			continue;
		}
	}
	return 0;
}

