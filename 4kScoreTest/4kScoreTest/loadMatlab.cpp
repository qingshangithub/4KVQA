#include<iostream>
#include <stdio.h>
#include <fstream>
#include <String>
using namespace std;

int main() {
	ifstream fin("data.txt");
	int dct_data[241][382];
	for (int i = 0; i < 241; i++)
	{
		for (int j = 0; j < 382; j++) {
			dct_data[i][j] = 0;
		}
	}
	int tmpi = 0, tmpj = 0,flag=0;
	while (fin)
	{
		char tmp;
		fin >> noskipws;
		fin >> tmp;

		if (tmp == '\"'&&flag == 0) {
			flag = 1;
			continue;
		}
		//if (tmp == '\"'&&flag == 1) {
		//	flag = 0;
		//	tmpj = 0;
		//	tmpi++;
		//	continue;
		//}
		if (tmp == ' '||tmp=='\n')continue;
		if (tmp>='0'&&tmp<='9')
		{
			int eachdata = tmp-'0';
			char tmp2;
			do {
				fin >> noskipws;
				fin >> tmp2;
				if (tmp2 == '\n')break;
				if (tmp2 == '\"')
				{
					flag = 0;
					break;
				}
				int addData = tmp2 - '0';
				eachdata = eachdata * 10 + addData;
			} while (1);
			if (flag != 0) {
				dct_data[tmpi][tmpj++] = eachdata;
			}
			else {
				dct_data[tmpi][tmpj++] = eachdata;
				tmpj = 0;
				tmpi++;
			}
		}
		
	}
	cout << dct_data[1][10];
	return 0;
}