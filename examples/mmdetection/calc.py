while True:
    print("식 입력 ('종료' 입력 시 종료)")
    line = input(">>> ")

    if line == '종료':
        break
    
    try:
        result = eval(line)
    except:
        print("똑바로 입력해라", end='\n\n')
        continue
    
    if not isinstance(result, (int, float)):
        print("똑바로 입력해라", end='\n\n')
        continue

    print("결과:", result, end='\n\n')
