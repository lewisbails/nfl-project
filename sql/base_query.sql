select
    p.pid, fg.fkicker, fg.good, fg.dist,
    g.seas as year, k.seas as seasons,
    case when (g.temp>50) or (g.temp is null) or (g.temp = '') then 0 else 1 end as cold,
    case when g.stad like "%Mile High%" then 1 else 0 end as altitude,
    case when (g.humd<60) or (g.humd = '') or (g.humd is null) then 0 else 1 end as humid,
    case when (g.wspd<10) or (g.wspd is null) or (g.wspd = '') then 0 else 1 end as windy,
    case when g.v=p.off then 1 else 0 end as away_game,
    case when g.wk>17 then 1 else 0 end as postseason,
    case when (pp.qtr=p.qtr) and ((pp.timd-p.timd)>0 or (pp.timo-p.timo)>0) then 1 else 0 end as iced,
    case g.surf when 'Grass' then 0 else 1 end as turf,
    case when g.cond like "%Snow%" then 1 when g.cond like "%Rain%" and not "Chance Rain" then 1 else 0 end as precipitation,
    case when p.qtr=4 and ABS(p.ptso - p.ptsd)>21 then 0
when p.qtr=4 and p.min<2 and ABS(p.ptso - p.ptsd)>8 then 0
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd < -7 then 0
when p.qtr<=3 then 0
when p.qtr=4 and p.min>=2 and ABS(p.ptso - p.ptsd)<21 then 0
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=5 and p.ptso-p.ptsd <=8 and fg.fgxp='FG' then 0 -- FG sealer
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=7 and p.ptso-p.ptsd <=9 and fg.fgxp='XP' then 0 -- XP sealer
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=-4 and p.ptso-p.ptsd <=-6 and fg.fgxp='FG' then 0 -- FG to bring within fg
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=-3 and p.ptso-p.ptsd <=-5 and fg.fgxp='XP' then 0 -- XP to bring within fg
else 1 end as pressure
from FGXP fg
    left join PLAY p on fg.pid=p.pid
    left join game g on p.gid=g.gid
    join kicker k on k.player = fg.fkicker and g.gid=k.gid
    join PLAY pp on pp.pid=p.pid-1 and pp.gid=p.gid