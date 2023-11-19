TimeBankPT
----------
Document APW19980227.0494.tml
Removed space in event with eid="e32": 'exigiria ' -> 'exigiria'

AQUAINT (TempEval3)
-------------------
Document APW20000115.0209 APW20000107.0088
These files contain the same text but differ on the document creation time.
The APW20000107.0088 was removed from the train corpus.

TimeBank (TempEval3)
--------------------
Document AP900816-0139
Event e3000 missing MAKEINSTANCE.
Added <MAKEINSTANCE eid="e3000" eiid="ei3000" tense="NONE" aspect="NONE" polarity="POS" pos="VERB" />

Document APW19980213.1310
Event e3000 missing MAKEINSTANCE.
Added <MAKEINSTANCE eid="e3000" eiid="ei3000" tense="NONE" aspect="NONE" polarity="POS" pos="VERB" />

Document ea980120.1830.0071
Event e3000 missing MAKEINSTANCE.
Added <MAKEINSTANCE eid="e3000" eiid="ei3000" tense="NONE" aspect="NONE" polarity="POS" pos="VERB" />

Document ea980120.1830.0071
Event e3001 missing on original document.
Added <MAKEINSTANCE eid="e3001" eiid="ei3001" tense="NONE" aspect="NONE" polarity="POS" pos="OTHER" />

TempEval2 French
----------------
Added offsets to each entity.

Moved document creation time TIMEX3 out of the TEXT tag.

Changed function_in_document="PUBLICATION_TIME" to function_in_document="CREATION_TIME".

Removed relations on file baldwin_frratrain_15.xml since the event "ei14" does no exist.
<TLINK from="ei14" lid="l19" origin="USER" relType="OVERLAPS" to="e15" />
<SLINK from="ei4" lid="l10" origin="USER" relType="MODAL" subordinatedEventInstance="ei5" />

Formatted dates for documents bove.xml, baldwin_frratrain_15.xml, and algerie.xml

Replaced value of DCT in bio_butler.xml from "PRESENT_REF" to "2003-01-15" (this date is consistent with the tlink annotations)


MeanTime (all languages)
------------------------

Document 120578_Automobile_sales_in_the_United_States_down_sharply.xml replaced
<TIMEX3 anchorTimeID="" beginPoint="" comment="" endPoint="" functionInDocument="CREATION_TIME" m_id="14" type="DATE" value="03-02-2009">
by
<TIMEX3 anchorTimeID="" beginPoint="" comment="" endPoint="" functionInDocument="CREATION_TIME" m_id="14" type="DATE" value="2009-02-03">

ENGLISH Corpus

Document 8951_World_largest_passenger_airliner_makes_first_flight replaced
<TIMEX3 anchorTimeID="29" beginPoint="" comment="" endPoint="" functionInDocument="NONE" m_id="29" type="DATE" value="2005-04-27">
by
<TIMEX3 anchorTimeID="29" beginPoint="" comment="" endPoint="" functionInDocument="CREATION_TIME" m_id="29" type="DATE" value="2005-04-27">

Document 8951_World_largest_passenger_airliner_makes_first_flight replaced
<TIMEX3 anchorTimeID="" beginPoint="" comment="" endPoint="" functionInDocument="NONE" m_id="46" type="DATE" value="2007-09-01">
by
<TIMEX3 anchorTimeID="" beginPoint="" comment="" endPoint="" functionInDocument="CREATION_TIME" m_id="46" type="DATE" value="2007-09-01">

File 8951 removed the following lines:
    <token number="12" sentence="2" t_id="13">The</token>
    <token number="13" sentence="2" t_id="14">A380</token>
    <token number="14" sentence="2" t_id="15">was</token>
    <token number="15" sentence="2" t_id="16">revealed</token>
    <token number="16" sentence="2" t_id="17">in</token>
    <token number="17" sentence="2" t_id="18">January</token>
    <token number="18" sentence="2" t_id="19">2005</token>

DUTCH corpus

deleted file 60658_DaimlerChrysler_plans_to_cut_13,000_jobs.xml
renamed 60658_DaimlerChrysler_plans_to_cut_13,000_jobs_dutch_utf8.xml to 60658_DaimlerChrysler_plans_to_cut_13,000_jobs.xml

Spanish
Fixed 114864_Global_markets_plunge.naf file: added date in the second line and removed unknown characters.

Italian
Fixed raw text of file 102977_Airbus_parent_EADS_wins_13_billion_UK_RAF_airtanker_contract as it missmatched with the annotated corpus
Same for file 101354
60658
279494