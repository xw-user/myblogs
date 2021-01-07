```sql

```

# 单行函数

```sql
-- 特性
-- 单行函数对单行操作
-- 每行返回一个结果
-- 有可能返回值与原参数数据类型不一致
-- 单行函数可以写在SELECT、WHERE、ORDER BY子句中
-- 有些函数没有参数，有些函数包括一个或多个参数
-- 函数可以嵌套

-- 数学
select PI()
select RAND()*10
-- 向上取整
select ceil(5.1)
-- 向下取整
select floor(6.6)
-- 四舍五入
SELECT ROUND(5.1789,2)
-- 截断
select TRUNCATE(5.1789,2)

-- 字符串函数
select CHAR_LENGTH('我23')
select LENGTH('我23')
select LENGTH('我')
-- 连接
select CONCAT('A','B')
-- 插入字符串 insert(原字符串,位置,长度,插入的字符串)
select insert('1234',1,0,'aaa')
-- 大写
select UPPER('abcD')
-- 小写
select lower('abcD')

select lower(ename) from emp

-- 返回字符串
-- 返回最左边指定长度的内容
select LEFT('123456',3)
-- 返回最右边指定长度的内容
select right('123456',3)
-- 填充
-- 左填充
select LPAD('123',10,'!')
-- 右填充
select RPAD('123',10,'!')
-- 删除空格
select ltrim('     123')
select length(Rtrim('123      '))
select length(rtrim(ltrim('    123   ')))
-- 上面这个去除首尾的空格相当于下面这个
select length(trim('   123    '))

-- trim  默认去除首尾的空格 但是 下面这个写法 去除首位指定的字符串
select TRIM('!' from '!!!wwww!!!!')
select TRIM(LEADING '!' from '!!!wwww!!!!')
select TRIM(TRAILING '!' from '!!!wwww!!!!')

-- 重复
select repeat('123',4)
-- 替换
select REPLACE('waw','w','qq')
-- 截取
select SUBSTRING('123456',2,3)


-- 写一个查询,用首字母大写，其它字母小写显示雇员的 ename，显示名字的长度，
-- 并给每列一个适当的标签，条件是满足所有雇员名字的开始字母是J、A 或 M 的雇员，
-- 并对查询结果按雇员的ename升序排序。（提示：使用length、substr）
SELECT CONCAT(UPPER(SUBSTRING(ename,1,1)),Lower(SUBSTRING(ename,2))) 员工姓名,LENGTH(ename) 名字的长度
FROM emp
WHERE SUBSTRING(ename,1,1) in ('J','A','M')
order BY ename 
-- 1
select UPPER(SUBSTRING(ename,1,1)) from emp
-- 2
select Lower(SUBSTRING(ename,2)) from emp

-- 查询员工姓名中中包含大写或小写字母A的员工姓名
SELECT ename
FROM emp
WHERE ename like '%A%' or  ename like '%a%'

-- 日期时间
select SYSDATE()
-- 系统日期
select CURDATE()
-- 日期时间
select now()
-- 时间
SELECT CURTIME()

-- 日期时间的 运算
-- 返回两个日期之间的时间差  注意 参数是 日期和时间的结合  必要要有时间
select TIMEDIFF('2020-12-2 12:00:00','2020-12-2 11:00:00')

-- 返回两个日期之间的天数差
select DATEDIFF('2020-12-2','2020-11-2')

-- 加上一个时间间隔
select DATE_ADD('2020-12-2',INTERVAL 7 day)
select DATE_ADD('2020-12-2 12:00:00',INTERVAL 30 MINUTE)

-- 减去一个时间间隔
select DATE_SUB('2020-12-2',INTERVAL 2 day)
select DATE_SUB('2020-12-2 12:00:00',INTERVAL 1 week)

-- 修改部门20的员工信息，把82年之后入职的员工入职日期向后调整10天
update emp set hiredate=DATE_ADD(hiredate,INTERVAL 10 day) 
where deptno=20 and hiredate>'1982-1-1'

-- 取出日期中单独的部分
-- 获取系统时间的年份 月份 日、
-- 方法1
select YEAR(CURDATE()),MONTH(CURDATE()),DAY(CURDATE())

-- 方法2
select EXTRACT(YEAR_MONTH FROM CURDATE())

-- 格式化
-- 2020年12月2日
select DATE_FORMAT(now(),'%Y年%m月%d日的%H点%i分%s秒')
-- 时间格式化 TIME_FORMAT(time,format)

-- 日期在一周中是第几天  美国制
select DAYOFWEEK('2020-12-2')
select DAYOFMONTH('2020-12-2')
select DAYOFYear('2020-12-2')

-- 2.查询部门10,20的员工截止到2000年1月1日，工作了多少个月，入职的月份。
SELECT CEIL(DATEDIFF('2000-1-1',hiredate)/30) 工作的月数,MONTH(hiredate)
FROM emp
WHERE deptno in(10,20)

-- 3.如果员工试用期6个月，查询职位不是MANAGER的员工姓名，
-- 入职日期，转正日期，入职日期是第多少月，第多少周
SELECT hiredate 入职,DATE_ADD(hiredate,INTERVAL 6 month) 转正,month(hiredate),WEEKOFYEAR(hiredate)
FROM EMP
where job !='MANAGER'


-- 流程控制函数
-- case when  then  end
select case when sal>2500 then 'good' when sal<1000 then 'low'  end from emp



```

# 多行函数

```sql
-- IF 流程控制语句
select if(1>2,"ok","no")
-- 其他
select PASSWORD('123456')
select MD5('123456')

-- 笛卡儿积
select * from emp,dept

-- where  n张 n-1链接条件
-- 等值连接
-- 姓名 部门名称 工作地点
SELECT ename,dname,loc
FROM emp,dept
WHERE emp.DEPTNO=dept.DEPTNO

-- 限制歧义列名 表别名
-- 表名 别名
-- 工作地点在芝加哥的员工的姓名 部门编号 部门名称
SELECT ename,e.deptno,loc
FROM emp e,dept d
WHERE e.DEPTNO=d.DEPTNO and loc='CHICAGO'

-- 多表连接写法
-- 1 分析列 来自那些表 from xxxx
-- 2 分析表之间的关系 如果两张表之间有关系是最好的
--    如果没有关系，肯定会有一个中间表，需要在from中补充这个中间表 
-- 3 写关联条件  记得n张 n-1条件
-- 4 分析 其他的查询条件 where 语句完成
-- 5 select语句 查哪些内容
-- 6 有没有排序 有就写 没有就不写

-- 非等值连接
-- 查询员工的姓名 薪资以及薪资等级
SELECT ename,SAL,grade
FROM emp e,salgrade s
WHERE sal BETWEEN losal and hisal 

-- 查询每个员工的编号，姓名，工资，工资等级，所在工作城市，按照工资等级进行升序排序。
SELECT empno,ename,sal,grade,loc
FROM emp e,dept d,salgrade s
WHERE e.deptno=d.deptno and (e.sal BETWEEN losal and hisal)
order by grade

-- 自身链接
-- 每个员工的姓名和直接上级姓名
SELECT e.ename 员工,m.ename 经理
FROM emp e , emp m
-- 员工表的经理编号和经理表中的员工编号是一样的 只不过 员工表和经理表是一张表
WHERE e.mgr=m.empno


-- 1.查询所有工作在NEW YORK和CHICAGO的员工姓名，员工编号，以及他们的经理姓名，经理编号。
SELECT e.empno,e.ename 员工,m.empno,m.ename 经理
FROM emp e , emp m,dept d
WHERE e.mgr=m.empno and e.deptno=d.deptno and loc in('NEW YORK','CHICAGO')

-- sql99
-- 交叉连接 进了解 是无效的连接
SELECT * from emp 
CROSS JOIN dept


-- 自然连接   等值连接  列名相同 数据类型
select * from emp NATURAL join dept

-- 指定 用哪一个列或者几个列作为连接条件  列名数据类型相同
-- using
SELECT *
FROM emp join dept using(deptno)
-- 注意  using中不能使用表名或者别名作为列名的前缀
-- 自然连接和using不能同时使用

-- on 
-- 指定 连接条件 或者指定要连接的列 可以用on
-- on将 连接条件和条件查询的条件分隔开的  提高可读性
select * from emp e join dept d on e.deptno=d.deptno


-- 1.查询所有工作在NEW YORK和CHICAGO的员工姓名，员工编号，以及他们的经理姓名，经理编号。
SELECT e.empno,e.ename 员工,m.empno,m.ename 经理
FROM emp e , emp m,dept d
WHERE e.mgr=m.empno and e.deptno=d.deptno and loc in('NEW YORK','CHICAGO');

SELECT e.empno,e.ename 员工,m.empno,m.ename 经理
FROM emp e join emp m on  e.mgr=m.empno join  dept d on e.deptno=d.deptno
WHERE loc in('NEW YORK','CHICAGO');

-- 查询每个员工的编号，姓名，工资，工资等级，所在工作城市，按照工资等级进行升序排序。
SELECT empno,ename,sal,grade,loc
FROM emp e,dept d,salgrade s
WHERE e.deptno=d.deptno and (e.sal BETWEEN losal and hisal)
order by grade;

SELECT empno,ename,sal,grade,loc
FROM emp e join dept d on e.deptno=d.deptno join  salgrade s on (e.sal BETWEEN losal and hisal)
order by grade;

-- 查询 员工姓名 部门名称 包括没有部门的员工也要显示出来
-- 左外连接
select ename,dname
from emp e LEFT outer JOIN dept d
on e.DEPTNO=d.DEPTNO
-- 右外连接
select ename,dname
from  dept d right outer JOIN emp e
on e.DEPTNO=d.DEPTNO

-- 1.查询部门编号 部门名称 员工姓名 包括没有员工的部门也要显示
select d.deptno,dname,ename
from  dept d left outer JOIN emp e
on e.DEPTNO=d.DEPTNO
-- 2.使用自然连接，显示入职日期在80年5月1日之后的员工姓名，部门名称，入职日期
SELECT ename,dname,hiredate
FROM emp NATURAL JOIN dept
WHERE HIREDATE>'1980-5-1'
-- 3.使用USING子句，显示工作在CHICAGO的员工姓名，部门名称，工作地点
SELECT ename,dname,loc
FROM emp join dept USING(deptno)
WHERE loc='CHICAGO'
-- 4.使用ON子句，显示工作在CHICAGO的员工姓名，部门名称，工作地点，薪资等级
SELECT ename,dname,grade,loc
FROM emp e join dept d on e.deptno=d.deptno join  salgrade s on (e.sal BETWEEN losal and hisal)
WHERE loc='CHICAGO'
-- 5.使用左连接，查询每个员工的姓名，经理姓名，没有经理的King也要显示出来。
SELECT e.ename,m.ename 经理
FROM emp e left OUTER JOIN emp m
on e.mgr=m.empno
-- 6.使用右连接，查询每个员工的姓名，经理姓名，没有经理的King也要显示出来。
SELECT e.ename,m.ename 经理
FROM emp m right OUTER JOIN emp e
on e.mgr=m.empno

```

# 高级查询

```sql
-- 高级查询
-- 分组函数
-- 单行函数 一行得到一个结果  多行函数 多行得到一个结果
-- 多行函数包括 分组函数 聚合函数
-- 需要先将数据按照一定的方式分组，进行查询数据。将数据分组的函数就是分组函数
-- 分组函数：将多行数据进行分组处理并得到一个处理的结果
-- 常用的分组函数  min() max()  sum() avg() count()
-- 查询emp表中薪资最高是多少
select max(sal),min(sal),sum(sal),avg(sal)
from emp

-- 查询emp表中奖金最高 最小 总和 平均
-- 分组函数中 可以嵌套对null的处理
select max(comm),min(ifnull(COMM,0)),sum(ifnull(COMM,0)),avg(ifnull(COMM,0))
from emp

-- 查询 员工 数量
select count(empno)
from emp

-- 查询 emp中 部门的个数
-- 分组函数和DISTINCT的结合
select count(DISTINCT deptno)
from emp




```

# 高级查询-子查询

```sql
-- 高级查询
-- 分组查询
-- 处理问题的时候 出现按照一定规则（条件）分组
-- 然后在使用分组函数对已经分好组的数据进行处理
-- 不仅需要分组函数 还需要 进行按照规则分组（group by 进行数据分组）
-- select 列名 from 表名 where字句 group by 分组列

-- 注意
-- group by 是将where是筛选过后的符合条件的数据 在进行分组操作
-- group by中分组依据的字段 可以不在select中出现
-- 在SELECT列表中除了分组函数那些项，所有列都必须包含在GROUP BY 子句中

-- 查询每个部门的编号 平均薪资
select deptno,avg(sal)
FROM emp
group by deptno
-- 查询不同岗位的平均薪资和员工人数
SELECT avg(sal),count(empno),ename
FROM emp
group BY job
-- 可以出现多个分组依据列
-- 查询 每个部门每个岗位的薪资总和
SELECT deptno,job,sum(sal)
FROM emp
GROUP BY deptno,job 

-- 1.查询每个部门的部门编号，部门名称，部门人数，最高工资，最低工资，工资总和，平均工资。
SELECT e.deptno,dname,count(e.empno), max(sal),min(sal),sum(sal),avg(sal)
FROM emp e,dept d
WHERE e.deptno=d.deptno
GROUP BY e.deptno,dname
-- 2.查询每个部门，每个岗位的部门编号，部门名称，岗位名称，部门人数，最高工资，最低工资，工资总和，平均工资。
SELECT e.deptno,dname,count(e.empno), max(sal),min(sal),sum(sal),avg(sal)
FROM emp e,dept d
WHERE e.deptno=d.deptno
GROUP BY e.deptno,dname,job
-- 3.查询每个经理所管理的人数，经理编号，经理姓名，要求包括没有经理的人员信息。
SELECT jl.empno,jl.ename,count(yg.empno)
FROM emp yg LEFT OUTER JOIN emp jl on yg.mgr=jl.empno
GROUP BY jl.empno

-- 3.查询每个经理所管理的人数>=3，经理编号，经理姓名，要求包括没有经理的人员信息。
-- 分组查询的之后根据一些条件筛选结果 having
-- where having 区别
-- where实在groupby之前 having实在之后
-- group by 子句是在where子句筛选之后的满足条件的数据中继续分组
-- having 子句是在通过groupby 分组之后 对分组的结果进行筛选
-- 书写  select from where GROUP BY having order by
SELECT jl.empno,jl.ename,count(yg.empno)
FROM emp yg LEFT OUTER JOIN emp jl on yg.mgr=jl.empno
GROUP BY jl.empno
HAVING count(yg.empno)>=3
ORDER BY count(yg.empno)

-- 查询 不同部门的部门名称 平均薪资 最高薪资 人数，部门平均薪资不低于2000
SELECT d.deptno,dname,avg(sal),max(sal),count(*)
from emp e,dept d
where e.deptno=d.deptno
GROUP by d.deptno,dname
having avg(sal)>2000
-- 执行  from where group by having select order by

-- 2.查询部门平均工资大于2000，且人数大于2的部门编号，
-- 部门名称，部门人数，部门平均工资，并按照部门人数升序排序。
SELECT e.deptno,dname,count(e.empno),avg(sal)
FROM emp e,dept d
WHERE e.deptno=d.deptno
GROUP BY e.deptno,dname
HAVING count(e.empno)>2 and avg(sal)>2000
ORDER BY count(e.empno)

-- 查询薪资比ALLEN高的员工信息
select * from emp where sal>1600
select sal from emp where ename='ALLEN'

-- 子查询
-- 实现查询的时候 如果某查询的条件需要借助于另外一个查询的查询结果 此时需要子查询
-- 子查询 内部查询 内部选择  而包含子查询的语句被称作 外部查询 外部选择

-- 子查询的用法 子查询需要写在一对() 中，可以用在表达式的任何地方
-- 基本语法
-- select 列名 from 表名 where 列名 操作符 (select 列名 from 表名 where子句) 

-- 注意： 
-- 可以理解 子查询的结果作为操作符右侧的常量进行使用
-- 需要在括号中
-- 运算符的右边
-- 子查询 可以用在where having from 中  
-- 子查询中一般不用在order by中

-- 查询薪资比ALLEN高的员工信息
select * from emp where sal>(select sal from emp where ename='ALLEN')

-- 子查询 从结果上分
-- 单行子查询 结果单行单列
-- 多行子查询 结果多行单列
-- 多列子查询 结果多列

-- 单行子查询

-- 子查询在 where
-- 查询最低工资的员工信息
-- 1 员工信息
select * from emp
-- 2 查询最低薪资
select min(sal) from emp 
-- 3
select * from emp where sal=(select min(sal) from emp )
-- 方法2
select * from emp order by sal LIMIT 1

-- 子查询写在了 from中
-- 查询部门平均薪资最高的部门编号 部门的平均薪资
-- 1 平均薪资
select deptno,avg(sal) from emp GROUP BY deptno
-- 2 部门最高的薪资
select max(sal) from emp GROUP BY deptno
-- 此时发现emp中缺少我要的平均薪资 所以可以将子查询的结果作为一张表
SELECT result.deptno,max(pjsal)
FROM (select deptno,avg(sal) pjsal from emp GROUP BY deptno) result
-- 可以不用子查询 如下
select avg(sal) rs,deptno from emp group by deptno order by avg(sal) desc limit 1


-- 查询 部门编号，部门人数 ，部门的最低薪资 以及最低薪资的员工的姓名
-- 1查询部门的最低薪资
select deptno,min(sal) from emp GROUP BY deptno
-- 
SELECT result.deptno,result.low,result.rs,ename
FROM (select deptno,min(sal) low,count(*) rs  from emp GROUP BY deptno) result,emp e
WHERE result.deptno=e.deptno and e.sal=result.low





```

