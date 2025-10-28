# from TableOfContentExtractor import extract_course_reading_lists
# from ExcelExporter import export_to_excel
from Mapper import mapping_excel

if __name__ == "__main__":
    
    pdf_file = "./input/Module_Handbook_Physics.pdf" 
    mapping_file = "../mapper_excel/Course-Vi-unique.xlsx"
    target_file = "../output/course_reading_lists.xlsx"
    
    # extract raw reading list text from PDF
    # all_reading_lists = extract_course_reading_lists(pdf_file)
    
    # llm context recongnition and exporting to Excel
    # export_to_excel(all_reading_lists)

    # mapping vietnamese name
    mapping_excel(mapping_file, target_file)


    # for course in all_reading_lists:
    #     print(f"## {course['course_name']} ({course['course_id']})")
    #     print(course["reading_list"])
    #     print("-" * 25)