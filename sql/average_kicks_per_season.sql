select avg(x.kicks/x.kickers)
from
(select count(distinct(f.fkicker)) as kickers, count(*) as kicks, g.seas as year
from fgxp f
join play p on f.pid=p.pid
join game g on g.gid=p.gid
group by g.seas) x