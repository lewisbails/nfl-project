select 
p.pid,fg.good,fg.dist, 
g.seas as year, k.seas as seasons,
case when g.temp<50 then 1 else 0 end as cold,
case when g.stad like "%Mile High%" then 1 else 0 end as altitude,
case when g.humd>=60 then 1 else 0 end as humid,
case when g.wspd>=10 then 1 else 0 end as windy,
case when g.v=p.off then 1 else 0 end as away_game,
case when g.wk>=10 then 1 else 0 end as postseason,
case when pp.qtr=p.qtr then pp.timd-p.timd else 0 end as iced,
case g.surf when 'Grass' then 0 else 1 end as turf,
case when g.cond like "%Snow%" then 1 when g.cond like "%Rain%" and not "Chance Rain" then 1 else 0 end as precipitation,
case when p.qtr=4 and ABS(p.ptso - p.ptsd)>21 then 0
when p.qtr=4 and p.min<2 and ABS(p.ptso - p.ptsd)>8 then 0
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd < -7 then 0
when p.qtr<=3 then 0
when p.qtr=4 and p.min>=2 and ABS(p.ptso - p.ptsd)<21 then 0
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=5 and p.ptso-p.ptsd <=8 then 0
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=-4 and p.ptso-p.ptsd <=-6 then 0
else 1 end as pressure
from FGXP fg
left join PLAY p on fg.pid=p.pid
left join game g on p.gid=g.gid
join kicker k on k.player = fg.fkicker and g.gid=k.gid
join PLAY pp on pp.pid=p.pid-1 and pp.gid=p.gid
where fg.fgxp='FG' -- not an xp
and fg.fkicker in (
select kicker
from tenth_attempt) -- has had at least 10 attempts overall
and fg.pid > (
select pid
from tenth_attempt
where fg.fkicker = kicker) -- this kick came after the 10th attempt
order by p.pid