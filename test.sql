SELECT s."SP ID", s."Gender", s."Age", s."Work Experience/From Date", s."Work Experience/To Date", u."Work Experience/Tasks"
FROM structured_data s
JOIN unstructured_data u ON s."SP ID" = u."SP ID"
WHERE u."Work Experience/Tasks_embedding" <=> \'[...]\'
ORDER BY s."Work Experience/From Date" ASC LIMIT 5

SELECT sd."SP ID", sd."Gender", sd."Age", ud."Work Experience/Company", ud."Work Experience/Designation", ud."Skills" 
FROM structured_data sd 
JOIN unstructured_data ud ON sd."SP ID" = ud."SP ID" 
WHERE sd."Age" > 30 AND ud."Work Experience/Tasks_embedding" <=> '[query_vector_1]' < 0.6
ORDER BY sd."Age" DESC