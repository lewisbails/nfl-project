select
    p.pid, fg.fkicker, fg.good, fg.dist,
    g.seas as year, k.seas as seasons,
    g.temp as temperature,
    g.H as home_team,
    g.stad as stadium,
    k.team as team,
    case when fg.fgxp='XP' then 1 else 0 end as XP,
    round(datediff(str_to_date(concat('10/','01/',k.year), '%m/%d/%Y'), str_to_date(case when k.player='JH-0500' then '07/30/1976' when k.player='TP-1200' then '10/26/1970' else ppp.dob end, '%m/%d/%Y'))/365,1) as age,
    case when g.stad like "%Mile High%" and g.H like "%DEN%" then 1 else 0 end as altitude,
    case when (g.humd<60) or (g.humd = '') or (g.humd is null) then 0 else 1 end as humid,
    case when (g.wspd = '') or (g.wspd is null) then 0 else g.wspd end as wind,
    case when g.v=p.off then 1 else 0 end as away_game,
    case when g.wk>17 then 1 else 0 end as postseason,
    case when (pp.qtr=p.qtr) and ((pp.timd-p.timd)>0 or (pp.timo-p.timo)>0) then 1 else 0 end as iced,
    case g.surf when 'Grass' then 0 else 1 end as turf,
    case when g.cond like "%Snow%" then 1 when g.cond like "%Rain%" and not "Chance Rain" then 1 else 0 end as precipitation,
    -- pressure
    case when p.qtr=4 and ABS(p.ptso - p.ptsd)>21 then 0 -- game is basically over
when p.qtr<=3 then 1 -- normal field goal
when p.qtr=4 and p.min>=2 and ABS(p.ptso - p.ptsd)<21 then 2 -- early last qtr still potentially a game
when p.qtr=4 and p.min<2 and ABS(p.ptso - p.ptsd)>10 then 0 -- game is essentially over
-- tied
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd=0 then 5 -- XP or FG to win else OT
-- FG pressure
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=5 and p.ptso-p.ptsd <=10 and fg.fgxp='FG' then 2 -- FG sealer
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd<=4 and p.ptso-p.ptsd>=3 and fg.fgxp='FG' then 3 -- FG so opp needs TD
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd<=2 and p.ptso-p.ptsd>=1 and fg.fgxp='FG' then 4 -- FG so opp needs TD
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd=-1 or p.ptso-p.ptsd=-2 and fg.fgxp='FG' then 6 -- lose if miss else win
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd=-3 and fg.fgxp='FG' then 6 -- lose if miss else OT
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=-4 and p.ptso-p.ptsd <=-6 and fg.fgxp='FG' then 2 -- FG to bring within FG
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=-10 and p.ptso-p.ptsd <=-7 and fg.fgxp='FG' then 1 -- FG to bring within TD
-- XP pressure
when p.qtr=4 and p.min<2 and ABS(p.ptso - p.ptsd)>8 and fg.fgxp='XP' then 0 -- game is essentially over why they would go for 1 ill never know
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=7 and p.ptso-p.ptsd <=10 and fg.fgxp='XP' then 2 -- XP sealer
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd<=6 and p.ptso-p.ptsd>=5 and fg.fgxp='XP' then 3 -- XP so opp needs TD
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd<=4 and p.ptso-p.ptsd>=3 and fg.fgxp='XP' then 4 -- XP so opp needs TD
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd<=2 and p.ptso-p.ptsd>=1 and fg.fgxp='XP' then 4 -- XP so opp needs FG
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd=-1 and fg.fgxp='XP' then 6 -- lose if miss else OT
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd>=-4 and p.ptso-p.ptsd<=-2 and fg.fgxp='XP' then 2 -- XP to bring within fg not sure why theyd go XP when down 2 or 3
when p.qtr=4 and p.min<2 and p.ptso-p.ptsd>=-8 and p.ptso-p.ptsd>=-5 and fg.fgxp='XP' then 1 -- XP to bring within TD
-- OT pressure
when p.qtr>4 then 5
else 0 end as pressure
from FGXP fg
    left join PLAY p on fg.pid=p.pid
    left join game g on p.gid=g.gid
    join kicker k on k.player = fg.fkicker and g.gid=k.gid
    join PLAY pp on pp.pid=p.pid-1 and pp.gid=p.gid
    left join player ppp on k.player=ppp.player
where ((fg.fkicker in (select fkicker
    from fifty)) or (k.seas>=2 and k.seas-(g.seas-2000)>0))
    and p.blk != 1 -- blocked kicks are completely unpredictable and should not be counted