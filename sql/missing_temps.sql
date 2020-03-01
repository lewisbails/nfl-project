-- Missing temperatures and the average at that ground 

select g.gid, g.temp, g.H, g.stad, round(avg(case when gg.temp!='' then gg.temp else null end),1) as av
from game g 
join game gg on g.stad=gg.stad
where g.stad in (select distinct(ggg.stad) from game ggg where ggg.temp='')
group by g.gid
order by g.stad


-- select distinct(g.stad) from game g where g.temp='' 